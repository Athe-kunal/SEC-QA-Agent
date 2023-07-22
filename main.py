from LLM1 import get_response_llm1
from LLM2 import get_response_llm2
from load_database import download_vector_store_wandb,load_vector_store_local, get_relevant_docs,get_relevant_docs_via_mmr
from load_database import get_relevant_dict_with_mmr,get_relevant_docs_via_mmr
from utils import get_query_metadata
with open("user_request.txt","r") as f:
    USER_REQUEST = f.read()

restore_collection = load_vector_store_local("./sec-10-K-finbert","10-K",if_finbert=True)

def SEC_LLM(USER_REQUEST):
    # llm1_output_dict, query_metadata = get_response_llm1(USER_REQUEST,"10-K")
    llm1_output_dict = get_response_llm1(USER_REQUEST,"10-K")
    # llm1_output_dict = {'Section_Names': ['RISK FACTORS', 'MARKET RISK DISCLOSURES', 'LEGAL PROCEEDINGS', 'UNRESOLVED STAFF COMMENTS', 'MINE SAFETY'], 'Tickers': ['AAPL'], 'Years': ['2021', '2022'], 'augmented_query': ['Compare the risk associated with Apple stock for the year 2021', 'Compare the risk associated with Apple stock for the year 2022']}
    # query_metadata = get_query_metadata(llm1_output_dict)
    # relevant_sentences = get_relevant_docs(llm1_output_dict,query_metadata,restore_collection)
    relevant_dict = get_relevant_dict_with_mmr(llm1_output_dict,restore_collection,USER_REQUEST,if_finbert=True)
    # print(relevant_dict)
    relevant_sentences = get_relevant_docs_via_mmr(relevant_dict)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(relevant_sentences,USER_REQUEST,llm1_output_dict)
    # print(llm2_output)
    return llm2_output