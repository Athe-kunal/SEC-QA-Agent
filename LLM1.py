from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv
from langchain.llms import OpenAI
import os
import openai
from utils import get_query_metadata
from wandb.integration.openai import autolog
from config import *
section_names = ResponseSchema(
    name="Section_Names",
    description="Name of the sections"
)

tickers = ResponseSchema(
    name="Tickers",
    description="Name of the tickers, make sure to convert company names to tickers"
)
years = ResponseSchema(
    name="Years",
    description="Years mentioned, if no years are mentioned then output the last 5 years ['2018','2019','2020','2021','2022']"
)
augmented_query = ResponseSchema(
    name="augmented_query",
    description="Output the user query into list of sentences, where each sentence is a combination of a company name, year and query mentioned about it in the user query"
)

response_schema = [
    section_names,
    tickers,
    years,
    augmented_query
]

output_parse = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parse.get_format_instructions()

llm1_template = """
You are a financial statement analyst, and here are the definitions of different sections {definitions_of_sections} in a SEC filings.\n

The definition is formatted as Section_Name: definition of the section. Based on the definitions, return the top {num_returns} possible section names that the user is asking for.\n

User request: "{user_request}"
{format_instructions}
"""
'''
Return the stocks tickers mentioned in the user query, make sure that you convert the company name to ticker.\n

Return the years that are mentioned in the user query, and if nothing is mentioned, then output the last 5 years. \n
For example:
User Query: What are the risk factors for Apple and Google for the year 2021 and 2022?\n
Augmented Query: ["What are the risk factors for Apple in 2022?","What are the risk factors for Apple in 2021?","What are the risk factors for Google in 2022?","What are the risk factors for Google in 2021?"]\n
'''
llm_1_prompt_template =PromptTemplate(
    input_variables=["definitions_of_sections","user_request","num_returns","format_instructions"],
    template = llm1_template
)


# Load variables from .env file
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key
# autolog({"project":PROJECT, "job_type": JOB_TYPE+"_LLM1"})
# print(format_instructions)
def get_response_llm1(USER_REQUEST:str,filing_type:str="10-K")->dict:

    if filing_type=="10-K":
        section_def = DEFINITIONS_10K
    elif filing_type == "10-Q":
        section_def = DEFINITIONS_10Q
    llm1_prompt =  llm_1_prompt_template.format(
        definitions_of_sections=section_def,
        num_returns=NUM_SECTION_RETURN,
        user_request = USER_REQUEST,
        format_instructions=format_instructions
    )

    llm1 = OpenAI(temperature=0.0)

    output_1 = llm1.predict(llm1_prompt)
    llm1_output_dict = output_parse.parse(output_1)
    for key in llm1_output_dict:
        if not isinstance(llm1_output_dict[key],list):
            llm1_output_dict[key] = llm1_output_dict[key].split(", ") 
    llm1_output_dict['Section_Names'] = [i.upper() for i in llm1_output_dict['Section_Names']]
# print(llm1_output_dict)

    query_metadata = get_query_metadata(llm1_output_dict)
    return llm1_output_dict, query_metadata

