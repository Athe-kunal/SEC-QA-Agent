import wandb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
import chromadb
from collections import defaultdict
from utils import (
    key_modifier,
    modify_all_keys,
    get_query_metadata,
    get_relevant_sentences,
    maximal_marginal_relevance,
)
from config import NUM_RESULTS
from chromadb.utils import embedding_functions
import numpy as np
import itertools
import re

finbert_embed = HuggingFaceEmbeddings(model_name="ProsusAI/finbert")


class FinBertEmbeddings(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embed_out = finbert_embed.embed_documents(texts)
        return embed_out


def download_vector_store_wandb(
    wandb_run: wandb.run, doc_name: str, if_finbert: bool = False
):
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    # load vector store artifact
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config.vector_store_artifact, type="search_index"
    ).download()
    print(f"Downloaded vector store at {vector_store_artifact_dir}")
    # if if_finbert:
    #     embedding_fn = FinBertEmbeddings()
    # else:
    #     embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # chroma_restore_client = chromadb.Client(
    #     Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=vector_store_artifact_dir
    #     )
    # )
    # collection_name = f"SEC-{doc_name}"
    # restore_collection = chroma_restore_client.get_collection(name=collection_name,embedding_function=embedding_fn)
    # return restore_collection


def load_vector_store_local(
    vcectore_store_dir: str, doc_name, if_finbert: bool = False
):
    if if_finbert:
        embedding_fn = FinBertEmbeddings()
        chroma_restore_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=vcectore_store_dir
            )
        )
        collection_name = f"SEC-{doc_name}-finbert"
        restore_collection = chroma_restore_client.get_collection(
            name=collection_name, embedding_function=embedding_fn
        )
    else:
        chroma_restore_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=vcectore_store_dir
            )
        )
        collection_name = f"SEC-{doc_name}"
        restore_collection = chroma_restore_client.get_collection(name=collection_name)
    return restore_collection


def get_relevant_docs(query_metadata, restore_collection,user_request:str):
    # print(llm1_output_dict)
    if len(query_metadata) <= 1:
        where_clause = query_metadata[0]
    else:
        where_clause = {"$or": query_metadata}
    query_results = restore_collection.query(
        query_texts=user_request,
        n_results=20,
        where=where_clause,
        include=["metadatas", "documents", "distances", "embeddings"],
    )
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


def get_relevant_dict_with_mmr(
    llm1_output_dict,
    restore_collection,
    user_query,
    doc_name="10-K",
    if_finbert: bool = False,
):
    section_names = llm1_output_dict["Section_Names"]
    ticker_names = llm1_output_dict["Tickers"]
    years = llm1_output_dict["Years"]
    all_relevant_docs = defaultdict()
    if not if_finbert:
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        query_embed = default_ef(texts=user_query)
        query_embed_arr = np.array(query_embed[0])
    elif if_finbert:
        finbert = FinBertEmbeddings()
        query_embed = finbert(texts=user_query)
        query_embed_arr = np.array(query_embed[0])
    for tic in ticker_names:
        for yr in years:
            individual_metadata = [
                {
                    "full_metadata": elem[0]
                    + "_"
                    + elem[1]
                    + "_"
                    + elem[2]
                    + "_"
                    + doc_name
                }
                for elem in list(itertools.product([tic], [yr], section_names))
            ]
            if len(individual_metadata)==1:
                where_clause=individual_metadata[0]
            else:
                where_clause = {"$or": individual_metadata}
            query_results = restore_collection.query(
                query_texts=user_query,
                n_results=20,
                where=where_clause,
                include=["metadatas", "documents", "distances", "embeddings"],
            )
            idxs = maximal_marginal_relevance(
                query_embed_arr, query_results["embeddings"][0], k=10
            )
            relevant_docs = [query_results["documents"][0][i] for i in idxs]
            relevant_metadata = [query_results["metadatas"][0][i] for i in idxs]
            all_relevant_docs[f"{tic}_{yr}"] = {
                "docs": relevant_docs,
                "metadata": relevant_metadata,
            }

    return all_relevant_docs


def get_relevant_docs_via_mmr(relevant_docs_dict: dict):
    relevant_sentences = ""

    for key, value in relevant_docs_dict.items():
        tic, year = key.split("_")
        relevant_sentences += f"Relevant documents for {tic} in the year {year}: " + "\n" 
        curr_docs = value["docs"]
        # print(curr_docs)
        # if not isinstance(curr_docs,str):
        #     curr_docs = str(curr_docs)
        # curr_docs = re.sub("\xa0", " ",curr_docs)
        # curr_docs = re.sub("\t", " ",curr_docs)
        # relevant_sentences += "\n ".join(value["docs"])
        relevant_sentences += " ".join(value["docs"])
        relevant_sentences += "\n\n"

    return relevant_sentences
