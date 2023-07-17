from LLM1 import get_response_llm1
from LLM2 import get_response_llm2
from load_database import download_vector_store_wandb,load_vector_store_local, get_relevant_docs
from utils import get_query_metadata
with open("user_request.txt","r") as f:
    USER_REQUEST = f.read()

restore_collection = load_vector_store_local("./sec-10-K","10-K",if_finbert=False)

def SEC_LLM(USER_REQUEST):
    llm1_output_dict, query_metadata = get_response_llm1(USER_REQUEST,"10-K")
    # print(llm1_output_dict)
    # llm1_output_dict = {'Section_Names': ['MARKET_FOR_REGISTRANT_COMMON_EQUITY', 'MANAGEMENT_DISCUSSION', 'MARKET_RISK_DISCLOSURES', 'FINANCIAL_STATEMENTS', 'ACCOUNTING_FEES'], 'Tickers': ['AAPL', 'GOOGL'], 'Years': ['2022'], 'augmented_query': ['How much is AAPL buying back shares for the year 2022?', 'How much is GOOGL buying back shares for the year 2022?']}
    # query_metadata = get_query_metadata(llm1_output_dict)
    relevant_sentences = get_relevant_docs(llm1_output_dict,query_metadata,restore_collection)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(relevant_sentences,USER_REQUEST,llm1_output_dict)
    # print(llm2_output)
    return llm2_output