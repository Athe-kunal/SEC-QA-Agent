import os
import fnmatch
import json
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.cache import SQLiteCache
from langchain.schema import Document
import wandb
import argparse
import langchain
from typing import List, Optional
from config import *
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


def find_files(directory, filename):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, filename):
            matches.append(os.path.join(root, filename))

    return matches

def get_input_files(directory,filing_type):
    file_names_list = []

    for root,_,filenames in os.walk(directory):
        for file in filenames:
            if file.startswith(filing_type):
                file_names_list.append(os.path.join(root,file))
    return file_names_list


def post_process(text):
    text = re.sub("\xa0"," ",text)
    text = re.sub(r"\\.", "", text)
    sentence_splits = text.split(".")
    sentence_with_delimiter = ".\n".join(sentence_splits)
    sentence_with_delimiter = re.sub(r" {2,}", "\n\n", sentence_with_delimiter)
    return sentence_with_delimiter


def load_documents(doc_name: str):
    files = find_files("data", f"{doc_name}.json")
    # files = get_input_files("data", doc_name)
    full_data = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
        full_data.append(data)

    documents = []
    metadata = []

    for tic_data in full_data:
        curr_year = tic_data["year"]
        ticker = tic_data["ticker"]
        filing_type = tic_data["filing_type"]

        for section, section_text in tic_data["all_texts"].items():
            documents.append(section_text)
            metadata.append(
                {
                    "year": curr_year,
                    "ticker": ticker,
                    "section": section,
                    "filing_type": filing_type,
                }
            )

    return documents, metadata


def chunk_documents(
    documents: List[str],
    metadata: List[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split each element in the list
    post_process_documents = [post_process(txt) for txt in documents]
    split_list = [
        text_splitter.split_text(element) for element in post_process_documents
    ]
    splitted_docs = []
    splitted_metadata = []

    for idx, docs in enumerate(split_list):
        curr_metadata = metadata[idx]
        if isinstance(docs, list):
            for doc in docs:
                splitted_docs.append(doc)
                splitted_metadata.append(curr_metadata)
        else:
            splitted_docs.append(docs)
            splitted_metadata.append(curr_metadata)

    post_process_splitted_metadata = []
    for idx, sm in enumerate(splitted_metadata):
        metadata_dict = {}
        metadata_dict.update(
            {
                "full_metadata": sm["ticker"]
                + "_"
                + sm["year"]
                + "_"
                + sm["section"]
                + "_"
                + sm["filing_type"]
            }
        )

        post_process_splitted_metadata.append(metadata_dict)
    assert len(splitted_docs) == len(
        post_process_splitted_metadata
    ), f"Length of splitted docs and metadata should be the same, but got {len(splitted_docs)} and {len(post_process_splitted_metadata)} respectively"

    all_splitted_doc = []

    for split_doc, split_meta in zip(splitted_docs, post_process_splitted_metadata):
        all_splitted_doc.append(Document(page_content=split_doc, metadata=split_meta))

    return all_splitted_doc, splitted_docs, post_process_splitted_metadata


finbert_embed = HuggingFaceEmbeddings(model_name="ProsusAI/finbert")


class FinBertEmbeddings(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embed_out = finbert_embed.embed_documents(texts)
        return embed_out


def create_vector_store_langchain(documents, doc_name: str, if_finbert: bool = False):
    if if_finbert:
        embedding_function = FinBertEmbeddings()
    else:
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        ids=[f"id{i}" for i in range(len(documents))],
        persist_directory=f"sec-{doc_name}",
        collection_name=f"SEC-{doc_name}",
    )
    vector_store.persist()
    return vector_store


def create_vector_store_chroma(
    splitted_docs,
    splitted_metadata,
    doc_name: str,
    if_delete: bool = False,
    if_finbert: bool = False,
):
    if if_finbert:
        collection_name = f"SEC-{doc_name}-finbert"
        persistent_directory = f"sec-{doc_name}-finbert"
    else:
        collection_name = f"SEC-{doc_name}"
        persistent_directory = f"sec-{doc_name}"

    chroma_client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet", persist_directory=persistent_directory
        )
    )

    # Vector database
    if if_delete:
        if len(chroma_client.list_collections()) > 0 and collection_name in [
            chroma_client.list_collections()[0].name
        ]:
            chroma_client.delete_collection(name=collection_name)
        # else:
    print(f"Creating collection: '{collection_name}'")
    if if_finbert:
        collection = chroma_client.create_collection(
            name=collection_name, embedding_function=FinBertEmbeddings()
        )
    else:
        collection = chroma_client.create_collection(name=collection_name)

    print("Building the vector database")

    collection.add(
        documents=splitted_docs,
        metadatas=splitted_metadata,
        ids=[f"id{i}" for i in range(1, len(splitted_docs) + 1)],
    )
    print(f"Completed building vectordatabase")
    chroma_client.persist()

    return chroma_client


def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb

    Args:
        vector_store_dir (str): The directory containing the vector store to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)


def main():
    parser = argparse.ArgumentParser(description="Document name to build database")
    parser.add_argument(
        "-doc",
        "--doc_name",
        type=str,
        default="10-K",
        help="Name of the SEC filings document",
    )
    parser.add_argument(
        "-fbert",
        "--finbert",
        type=bool,
        default=False,
        help="Name of the embeddings document",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmapps",
        type=str,
        help="The wandb project to use for storing artifacts",
    )

    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)
    doc_name = args.doc_name
    if_finbert = args.finbert
    langchain.llm_cache = SQLiteCache(database_path=f"sec-{doc_name}.db")

    documents, metadata = load_documents(doc_name)

    all_splitted_doc, splitted_docs, post_process_splitted_metadata = chunk_documents(
        documents, metadata
    )

    vector_store = create_vector_store_chroma(
        splitted_docs, post_process_splitted_metadata, doc_name, if_finbert=if_finbert
    )
    if if_finbert:
        log_index(f"./sec-{doc_name}-finbert", run)
    else:
        log_index(f"./sec-{doc_name}", run)
    run.finish()


if __name__ == "__main__":
    main()
