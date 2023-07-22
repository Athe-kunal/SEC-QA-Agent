from sec_filings import SECExtractor
import time
import json
import concurrent.futures
from collections import defaultdict
import os

import yaml

with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

assert data["document_type"] in [
    "10-K",
    "10-Q",
], "The supported document types are 10-K and 10-Q"

tickers = data["tickers"]
amount = data["amount"]
filing_type = data["document_type"]
num_workers = data["num_workers"]

se = SECExtractor(tickers, amount, filing_type)

os.makedirs("data", exist_ok=True)


def multiprocess_run(tic):
    # print(f"Started for {tic}")
    tic_dict = se.get_accession_numbers(tic)
    text_dict = defaultdict(list)
    for tic, fields in tic_dict.items():
        os.makedirs(f"data/{tic}", exist_ok=True)
        print(f"Started for {tic}")

        field_urls = [field["url"] for field in fields]
        years = [field["year"] for field in fields]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            results = executor.map(se.get_text_from_url, field_urls)
        for idx, res in enumerate(results):
            all_text, filing_type = res
            text_dict[tic].append(
                {
                    "year": years[idx],
                    "ticker": tic,
                    "all_texts": all_text,
                    "filing_type": filing_type,
                }
            )
    return text_dict


if __name__ == "__main__":
    start = time.time()
    thread_workers = min(len(tickers), num_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_workers) as executor:
        results = executor.map(multiprocess_run, tickers)

    # final_dict = []
    for res in results:
        # final_dict.append(res)
        curr_tic = list(res.keys())[0]
        for data in res[curr_tic]:
            # print(data)
            curr_year = data["year"]
            curr_filing_type = data["filing_type"]
            if curr_filing_type == "10-K":
                os.makedirs(f"data/{curr_tic}/{curr_year}", exist_ok=True)
                with open(
                    f"data/{curr_tic}/{curr_year}/{curr_filing_type}.json", "w"
                ) as f:
                    json.dump(data, f, indent=4)
            elif curr_filing_type == "10-Q":
                os.makedirs(f"data/{curr_tic}/{curr_year[:-2]}", exist_ok=True)
                with open(
                    f"data/{curr_tic}/{curr_year[:-2]}/{curr_filing_type}_{curr_year[-2:]}.json",
                    "w",
                ) as f:
                    json.dump(data, f, indent=4)

    print(f"It took {round(time.time()-start,2)} seconds")
