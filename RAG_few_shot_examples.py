from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain import hub
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.runnables import RunnablePassthrough
import pandas as pd 
import json

from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_excel(r"C:\Users\priyakedia\Desktop\Ticket examples.xlsx", engine = "openpyxl") #Replace it with the path to ticket file

df.drop(columns = ["Subcategory"], axis = 1, inplace = True)

llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo-16k", 
                      azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE"),
                      temperature=0.1, verbose=True)

embed_model = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002",
                                    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"))



# Index the few shot examples in vector db for contextual text classification
few_shot_docs = [
    Document(page_content=json.dumps({"Description" : df.iloc[i, 0], "Category" : df.iloc[i, 1]}))
    for i in range(51)
]

vector_db = FAISS.from_documents(few_shot_docs, embed_model)
retriever = vector_db.as_retriever(search_kwargs = {"k" : 5})

# examples = retriever.get_relevant_documents("6.money will not recived in idfc fist account merchant complaint id - idfc first", search_kwargs = {"k" : 3})
    
template = """Classify the text into one of the following categories :

Refund Requests
Payment Issues
Customer Support
UPI Payment Problems
Settlement Issues
Delivery/Service Issues
Technical Issues
Scam/Unauthorized Activity
Business/Transaction Inquiries

Detailed guidelines for how to classify:
Technical Issues : if the description is related to technical problem with the app or transaction not refelected in the dashboard
Payment Issues : If the customer complaints that the payment was not received
Settlement Issues : Questions related to settlements
Refund Requests : If the description says that the refund was not processed

Use the reference examples below to answer the question at the end. Make sure to confine the result in one of the above categories"

{examples}

Question: {description}

Category:"""

prompt = PromptTemplate.from_template(template)
    
def format_examples(docs):
    few_shot_examples = []
    for doc in docs:
        example = json.loads(doc.page_content)
        few_shot_examples.append({"Description" : example["Description"],
                                "Category" : example["Category"]})
    return "\n###EXAMPLE\n".join(json.dumps(example) for example in few_shot_examples)

# question = "118.hello razorpay team, please find the below mentioned transaction details of rs./*** is not reflected in the dashboard. kindly check and let us know the payment status. [image: ***] -- thanks, ***"   
# formatted_prompt = prompt.format(examples = format_examples(retriever.invoke(question)),
#                                  description = question)

rag_chain = (
    {"examples": retriever | format_examples, "description": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

classified_category = rag_chain.invoke("95.why not able to receive settlement .please help me to provide solution")

print(classified_category)

