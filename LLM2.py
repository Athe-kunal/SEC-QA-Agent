from langchain import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import OpenAI
import os
import openai
from wandb.integration.openai import autolog
from config import *
from langchain.chat_models import ChatOpenAI


# autolog({"project": PROJECT, "job_type": JOB_TYPE + "_LLM2"})


def get_response_llm2(relevant_sentences, user_query, llm1_output_dict):
    llm2_template = """
    You are a financial statement analyst with a strong understanding of financial documents and fundamental analysis. Base your answer only on the following relevant documentss: \n
    {relevant_documents} \n\n

    Answer the user question {user_query}\n

    Don't make up any information, and if the relevant information is not present, then just give the most similar answer to the user query from the relevant documents and politely give a warning that the information that the user is looking for, may not be in the documents\n
    
    Recheck your answer so that it is more coherent with what user is asking\n
    """

    llm2_prompt_template = PromptTemplate(
        input_variables=["relevant_documents", "user_query"], template=llm2_template
    )

    # user_query = llm1_output_dict["augmented_query"]
    # user_query = query_1+query_2
    llm2_prompt = llm2_prompt_template.format(
        user_query=user_query, relevant_documents=relevant_sentences
    )

    openai.api_key = os.environ["OPENAI_API_KEY"]

    llm_2 = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k",streaming=True)

    output = llm_2.predict(llm2_prompt)
    return output
