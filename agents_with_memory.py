from langchain.chat_models import AzureChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent, ConversationalChatAgent, initialize_agent
from langchain.tools import StructuredTool, BaseTool, Tool, tool
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai
from langchain.chains import LLMChain, LLMMathChain
from typing import Type, List, Dict, Any, Optional, Union, Tuple, Callable
from dotenv import load_dotenv
import os

os.getcwd()
load_dotenv("openai/End_to_end_Solutions/AOAISearchDemo/notebooks/db.env")

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

class CustomToolSchema(BaseModel):
    number1 : float = Field()
    number2 : float = Field()

# Custom schema for multiply utility
class CustomMultiplyTool(StructuredTool):
    name = "multiply"
    description = "Multiplies the two numbers. Input to the function should be two numbers that needs to be multiplied"
    args_schema : Type[BaseModel] = CustomToolSchema

    def _run(self, number1 : float, number2 : float) -> float:
        return number1*number2
    
    async def _arun(self, number1 : float, number2 : float) -> float:
        """Use the tool aynchronously"""
        raise NotImplementedError("custom tool doesn't support async")

# Custom schema for addition utility    
class CustomAddTool(StructuredTool):    
    name = "add"
    description = "Adds the two numbers. Input to the function should be two numbers that needs to be added"
    args_schema : Type[BaseModel] = CustomToolSchema

    def _run(self, number1 : float, number2 : float) -> float:    
        return number1+number2
    
    async def _arun(self, number1 : float, number2 : float) -> float:
        """Use the tool aynchronously"""
        raise NotImplementedError("custom tool doesn't support async")

# Custom schema for division utility    
class CustomDivideTool(StructuredTool):
    name = "dvide"
    description = "Divides the two numbers. Input to the function should be two numbers that needs to be divided"
    args_schema : Type[BaseModel] = CustomToolSchema

    def _run(self, number1 : float, number2 : float) -> float:
        return number1/number2
    
    async def _arun(self, number1 : float, number2 : float) -> float:     
        """Use the tool aynchronously"""
        raise NotImplementedError("custom tool doesn't support async")
    
llm_chat = AzureChatOpenAI(temperature=0, deployment_name= "priya-gpt-35-turbo", model = "gpt-35-turbo-0613")

# Tool list to keep a track of usable tools
tools = [CustomMultiplyTool(), CustomAddTool(), CustomDivideTool()]
#Agents are stateless, so adding memory to preserve chat history
memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, window_size=3)
memory.clear()
agent_kwargs = {"extra_prompt_messages" : [MessagesPlaceholder(variable_name="memory")]}
#initialize openai multi function agents with available tools and llm
agent = initialize_agent(tools, llm = llm_chat, agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
                         agent_kwargs=agent_kwargs, memory = memory, verbose = True)

query = "What is 5*5+10/10"
print(agent.run(query))
print(agent.run("What was the previous anser?"))
print(agent.run("Add 1 to the previous result"))
