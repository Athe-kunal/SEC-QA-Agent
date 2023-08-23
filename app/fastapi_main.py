from typing import Union

from fastapi import FastAPI
from app.LLM1 import get_response_llm1
from app.LLM2 import get_response_llm2
from app.load_database import download_vector_store_wandb,load_vector_store_local, get_relevant_docs,get_relevant_docs_via_mmr
from app.load_database import get_relevant_dict_with_mmr,get_relevant_docs_via_mmr
from app.utils import get_query_metadata
from pydantic import BaseModel
from typing import List
import os
import json

app = FastAPI()

restore_collection = load_vector_store_local("app/sec-10-K-finbert","10-K",if_finbert=True)

class llm1_output(BaseModel):
    Section_Names: List[str] = []
    Tickers: List[str] = []
    Years: List[str] = []
    user_query: str = ""
    
@app.get("/")
async def user_query(USER_REQUEST):
    llm1_output_dict = get_response_llm1(USER_REQUEST,"10-K")
    relevant_dict = get_relevant_dict_with_mmr(llm1_output_dict,restore_collection,USER_REQUEST,if_finbert=True)
    # print(relevant_dict)
    relevant_sentences = get_relevant_docs_via_mmr(relevant_dict)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(relevant_sentences,USER_REQUEST,llm1_output_dict)
    # print(llm2_output)
    return llm2_output

@app.get('/llm2/{llm1_dict}')
async def LLM2(llm1_str:str):
    # data_json = urllib.parse.unquote(llm1_dict)
    llm1_output_dict = json.loads(llm1_str)
    print(llm1_output_dict)
    relevant_dict = get_relevant_dict_with_mmr(llm1_output_dict,restore_collection,llm1_output_dict["user_query"],if_finbert=True)
    # print(relevant_dict)
    relevant_sentences = get_relevant_docs_via_mmr(relevant_dict)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(relevant_sentences,llm1_output_dict["user_query"],llm1_output_dict)
    # print(llm2_output)
    return llm2_output

@app.get("/full_text/{ticker}/{year}")
async def get_full_text_10k(ticker:str,year:str):
    for tic in os.listdir("data"):
        if tic == ticker:
            for yrs in os.listdir(f"data/{tic}"):
                if yrs == year:
                    with open(f'data/{tic}/{yrs}/10-K.json','r') as f:
                        data = json.load(f)
                    return data



