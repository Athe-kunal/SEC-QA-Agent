import itertools
from collections import defaultdict
from langchain.vectorstores import Chroma
import numpy as np
from typing import List
from langchain.math_utils import cosine_similarity
from requests.adapters import HTTPAdapter
from collections import namedtuple
from typing import List
from urllib3.util.retry import Retry
from pathlib import Path
from faker import Faker
import time
import requests
from datetime import date


def key_modifier(key):
    l = key.split("_")[:2]
    return l[0] + " (" + l[1] + ")"


def modify_all_keys(dictionary, key_modifier):
    modified_dict = {}
    for old_key, value in dictionary.items():
        new_key = key_modifier(old_key)
        modified_dict[new_key] = value
    return modified_dict


def get_query_metadata(llm1_output_dict, doc_name: str):
    section_names = llm1_output_dict["Section_Names"]
    ticker_names = llm1_output_dict["Tickers"]
    years = llm1_output_dict["Years"]

    query_metadata = [
        {"full_metadata": elem[0] + "_" + elem[1] + "_" + elem[2] + "_" + doc_name}
        for elem in list(itertools.product(ticker_names, years, section_names))
    ]

    return query_metadata


def get_relevant_sentences(query_results):
    all_relevant_sentences = defaultdict(str)
    for docs, metadatas in zip(query_results["documents"], query_results["metadatas"]):
        for each_doc, each_metadata in zip(docs, metadatas):
            key = each_metadata["full_metadata"]
            all_relevant_sentences[key] += each_doc

    relevant_dict = modify_all_keys(all_relevant_sentences, key_modifier)

    relevant_sentences = ""

    for key, value in relevant_dict.items():
        tic, year = key.split(" ")
        relevant_sentences += (
            f"Relevant documents for {tic} in the year {year[1:-1]} " + ": "
        )
        relevant_sentences += value
        relevant_sentences += "\n\n"

    return relevant_sentences


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


# Data Utils

MAX_RETRIES = 10
SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL = 0.1
FILING_DETAILS_FILENAME_STEM = "filing-details"
SEC_EDGAR_SEARCH_API_ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

retries = Retry(
    total=MAX_RETRIES,
    backoff_factor=SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,
    status_forcelist=[403, 500, 502, 503, 504],
)

FilingMetadata = namedtuple(
    "FilingMetadata",
    [
        "accession_number",
        "full_submission_url",
        "filing_details_url",
        "filing_details_filename",
    ],
)


class EdgarSearchApiError(Exception):
    pass


def form_request_payload(
    ticker_or_cik: str,
    filing_types: List[str],
    start_date: str,
    end_date: str,
    start_index: int,
    query: str,
) -> dict:
    payload = {
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "entityName": ticker_or_cik,
        "forms": filing_types,
        "from": start_index,
        "q": query,
    }
    return payload


def build_filing_metadata_from_hit(hit: dict) -> FilingMetadata:
    accession_number, filing_details_filename = hit["_id"].split(":", 1)
    # Company CIK should be last in the CIK list. This list may also include
    # the CIKs of executives carrying out insider transactions like in form 4.
    cik = hit["_source"]["ciks"][-1]
    accession_number_no_dashes = accession_number.replace("-", "", 2)

    submission_base_url = (
        f"{SEC_EDGAR_ARCHIVES_BASE_URL}/{cik}/{accession_number_no_dashes}"
    )

    full_submission_url = f"{submission_base_url}/{accession_number}.txt"

    # Get XSL if human readable is wanted
    # XSL is required to download the human-readable
    # and styled version of XML documents like form 4
    # SEC_EDGAR_ARCHIVES_BASE_URL + /320193/000032019320000066/wf-form4_159839550969947.xml
    # SEC_EDGAR_ARCHIVES_BASE_URL +
    #           /320193/000032019320000066/xslF345X03/wf-form4_159839550969947.xml

    # xsl = hit["_source"]["xsl"]
    # if xsl is not None:
    #     filing_details_url = f"{submission_base_url}/{xsl}/{filing_details_filename}"
    # else:
    #     filing_details_url = f"{submission_base_url}/{filing_details_filename}"

    filing_details_url = f"{submission_base_url}/{filing_details_filename}"

    filing_details_filename_extension = Path(filing_details_filename).suffix.replace(
        "htm", "html"
    )
    filing_details_filename = (
        f"{FILING_DETAILS_FILENAME_STEM}{filing_details_filename_extension}"
    )

    return FilingMetadata(
        accession_number=accession_number,
        full_submission_url=full_submission_url,
        filing_details_url=filing_details_url,
        filing_details_filename=filing_details_filename,
    )


fake = Faker()


def generate_random_user_agent() -> str:
    return f"{fake.first_name()} {fake.last_name()} {fake.email()}"


def get_filing_urls_to_download(
    filing_type: str,
    ticker_or_cik: str,
    num_filings_to_download: int,
    after_date: str,
    before_date: str,
    include_amends: bool,
    query: str = "",
) -> List[FilingMetadata]:
    """Get the filings URL to download the data

    Returns:
        List[FilingMetadata]: Filing metadata from SEC
    """
    filings_to_fetch: List[FilingMetadata] = []
    start_index = 0
    client = requests.Session()
    client.mount("http://", HTTPAdapter(max_retries=retries))
    client.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        while len(filings_to_fetch) < num_filings_to_download:
            payload = form_request_payload(
                ticker_or_cik,
                [filing_type],
                after_date,
                before_date,
                start_index,
                query,
            )
            headers = {
                "User-Agent": generate_random_user_agent(),
                "Accept-Encoding": "gzip, deflate",
                "Host": "efts.sec.gov",
            }
            resp = client.post(
                SEC_EDGAR_SEARCH_API_ENDPOINT, json=payload, headers=headers
            )
            resp.raise_for_status()
            search_query_results = resp.json()

            if "error" in search_query_results:
                try:
                    root_cause = search_query_results["error"]["root_cause"]
                    if not root_cause:  # pragma: no cover
                        raise ValueError

                    error_reason = root_cause[0]["reason"]
                    raise EdgarSearchApiError(
                        f"Edgar Search API encountered an error: {error_reason}. "
                        f"Request payload:\n{payload}"
                    )
                except (ValueError, KeyError):  # pragma: no cover
                    raise EdgarSearchApiError(
                        "Edgar Search API encountered an unknown error. "
                        f"Request payload:\n{payload}"
                    ) from None

            query_hits = search_query_results["hits"]["hits"]

            # No more results to process
            if not query_hits:
                break

            for hit in query_hits:
                hit_filing_type = hit["_source"]["file_type"]

                is_amend = hit_filing_type[-2:] == "/A"
                if not include_amends and is_amend:
                    continue
                if is_amend:
                    num_filings_to_download += 1
                # Work around bug where incorrect filings are sometimes included.
                # For example, AAPL 8-K searches include N-Q entries.
                if not is_amend and hit_filing_type != filing_type:
                    continue

                metadata = build_filing_metadata_from_hit(hit)
                filings_to_fetch.append(metadata)

                if len(filings_to_fetch) == num_filings_to_download:
                    return filings_to_fetch

            # Edgar queries 100 entries at a time, but it is best to set this
            # from the response payload in case it changes in the future
            query_size = search_query_results["query"]["size"]
            start_index += query_size

            # Prevent rate limiting
            time.sleep(SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL)
    finally:
        client.close()

    return filings_to_fetch
