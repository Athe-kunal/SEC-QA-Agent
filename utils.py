import itertools
from collections import defaultdict
from langchain.vectorstores import Chroma
import numpy as np
from typing import List
from langchain.math_utils import cosine_similarity

def key_modifier(key):
    l = key.split("_")[:2]
    return l[0]+" ("+l[1]+")"

def modify_all_keys(dictionary, key_modifier):
    modified_dict = {}
    for old_key, value in dictionary.items():
        new_key = key_modifier(old_key)
        modified_dict[new_key] = value
    return modified_dict

def get_query_metadata(llm1_output_dict,doc_name:str):
    section_names = llm1_output_dict['Section_Names']
    ticker_names = llm1_output_dict['Tickers']
    years = llm1_output_dict['Years']

    query_metadata = [{"full_metadata":elem[0]+"_"+elem[1]+"_"+elem[2]+"_"+doc_name} for elem in list(itertools.product(ticker_names,years,section_names))]

    return query_metadata

def get_relevant_sentences(query_results):
    all_relevant_sentences = defaultdict(str)
    for docs,metadatas in zip(query_results['documents'],query_results['metadatas']):
        for each_doc,each_metadata in zip(docs,metadatas):
            key = each_metadata["full_metadata"]
            all_relevant_sentences[key] += each_doc

    relevant_dict = modify_all_keys(all_relevant_sentences,key_modifier)

    relevant_sentences = ""

    for key,value in relevant_dict.items():
        tic,year = key.split(" ")
        relevant_sentences+=f"Relevant documents for {tic} in the year {year[1:-1]} "  + ": "
        relevant_sentences+= value
        relevant_sentences+="\n\n"
    
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
