from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from dotenv import load_dotenv
import json
import os

## Load the enviornment variables
load_dotenv()

# Load the document and convert it into markdowm format
loader = AzureAIDocumentIntelligenceLoader(file_path="data/input_data/2023_Annual_Report_test.pdf", api_key = os.getenv("form_recogniser_key"), api_endpoint = os.getenv("form_recogniser_endpoint"), 
                                            api_model="prebuilt-layout", mode = "markdown")
docs = loader.load()

 
# Split the document into chunks base on markdown headers.
# headers_to_split_on = [
#     ('<figure>', "Header 1"),
#     ('![](figures', 'Header 2'),
#     ('</figure>', "Header 3"),
# ]

# Update it based on the chunking strategy 
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("| |", "Header 4"),
    ('<figure>', "Header 5"),
    ('![](figures', "Header 6"),
    (r'</figure>', "Header 7")
]

text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

docs_string = docs[0].page_content
splits = text_splitter.split_text(docs_string)

print(splits)
