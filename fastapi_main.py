from typing import Union

from fastapi import FastAPI
from LLM1 import get_response_llm1
from LLM2 import get_response_llm2
from load_database import (
    download_vector_store_wandb,
    load_vector_store_local,
    get_relevant_docs,
    get_relevant_docs_via_mmr,
)
from load_database import get_relevant_dict_with_mmr, get_relevant_docs_via_mmr
from utils import get_query_metadata
from pydantic import BaseModel
from typing import List

app = FastAPI()

restore_collection = load_vector_store_local(
    "./sec-10-K-finbert", "10-K", if_finbert=True
)


class llm1_output(BaseModel):
    Section_Names: List[str]
    Tickers: List[str]
    Years: List[str]
    user_query: str


@app.get("/")
def user_query(USER_REQUEST):
    llm1_output_dict = get_response_llm1(USER_REQUEST, "10-K")
    relevant_dict = get_relevant_dict_with_mmr(
        llm1_output_dict, restore_collection, USER_REQUEST, if_finbert=True
    )
    # print(relevant_dict)
    relevant_sentences = get_relevant_docs_via_mmr(relevant_dict)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(relevant_sentences, USER_REQUEST, llm1_output_dict)
    # print(llm2_output)
    return llm2_output


@app.get("/llm2")
def LLM2(llm1_output_dict: llm1_output):
    relevant_dict = get_relevant_dict_with_mmr(
        llm1_output_dict, restore_collection, USER_REQUEST, if_finbert=True
    )
    # print(relevant_dict)
    relevant_sentences = get_relevant_docs_via_mmr(relevant_dict)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(
        relevant_sentences, llm1_output_dict.user_query, llm1_output_dict
    )
    # print(llm2_output)
    return llm2_output
