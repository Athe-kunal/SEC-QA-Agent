
import wandb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
import chromadb
from collections import defaultdict
from utils import *
from config import NUM_RESULTS
finbert_embed = HuggingFaceEmbeddings(model_name="ProsusAI/finbert")

class FinBertEmbeddings(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embed_out = finbert_embed.embed_documents(texts)
        return embed_out
    
def download_vector_store_wandb(wandb_run: wandb.run,doc_name:str,if_finbert:bool=False):
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

def load_vector_store_local(vcectore_store_dir:str,doc_name,if_finbert:bool=False):
    if if_finbert:
        embedding_fn = FinBertEmbeddings()
        chroma_restore_client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=vcectore_store_dir
            )
        )
        collection_name = f"SEC-{doc_name}"
        restore_collection = chroma_restore_client.get_collection(name=collection_name,embedding_function=embedding_fn)
    else:
        chroma_restore_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=vcectore_store_dir
            )
        )
        collection_name = f"SEC-{doc_name}"
        restore_collection = chroma_restore_client.get_collection(name=collection_name)
    return restore_collection

def get_relevant_docs(llm1_output_dict,query_metadata,restore_collection):
    # print(llm1_output_dict)
    if len(query_metadata)<=1:
        where_clause = query_metadata[0]
    else:
        where_clause = {"$or":query_metadata}
    query_results = restore_collection.query(
    query_texts=llm1_output_dict['augmented_query'],
    n_results=20//len(llm1_output_dict['augmented_query']),
    where=where_clause,
    include=["metadatas","documents","distances","embeddings"]
)
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
