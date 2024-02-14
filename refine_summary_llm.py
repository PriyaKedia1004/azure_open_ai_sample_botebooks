from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import LLMChain, StuffDocumentsChain, MapReduceDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain
import tiktoken
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chat_models import Azu
from langchain_community.document_loaders import WebBaseLoader
import json
from langchain.chains.summarize import load_summarize_chain

import random
import openai
import os

load_dotenv()

openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_type = "azure"

openai.api_version = "2023-12-01-preview"

llm = AzureChatOpenAI(azure_deployment=os.getenv("DEPLOYMENT_NAME"), api_version=os.getenv("OPENAI_API_VERSION"),
                      temperature = 0.01)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

def get_token_count(input_str):
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    token_length = len(tokenizer.encode(input_str))
    return token_length

## Load the data
docs = loader.load()

### Split the documents into chunks of 1000 tokens each chunk
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 5000, chunk_overlap = 0)
splitted_docs = text_splitter.split_documents(docs)

prompt_template = """Write a concise summary of the following:
{text}

CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_prompt = """Your job is to produce a final summary based on the existing summary upto a certain point : {existing_answer} with more context below
-------------
{text}
------------
If the context isn't useful, return the original summary.
Refined summary : 
"""

refine_template = PromptTemplate.from_template(refine_prompt)

chain = load_summarize_chain(llm = llm,
                             chain_type="refine",
                             question_prompt = prompt,
                             refine_prompt=refine_template,
                             return_intermediate_steps = True,
                             input_key = "input_documents",
                             output_key = "output_text")

result = chain({"input_documents" : splitted_docs}, return_only_outputs = True)

print(len(result["intermediate_steps"]))

print(result["intermediate_steps"][-1])

result["output_text"]
