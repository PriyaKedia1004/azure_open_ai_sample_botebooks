import requests
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv(".env")

search_url = "https://api.bing.microsoft.com/" + "v7.0/search"
subscription_key = os.getenv("BING_SUBSCRIPTION_KEY")

client = AzureOpenAI(azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                     api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                     api_version = os.getenv("OPENAI_API_VERSION"),
                     )

def get_search_results(search_query, n_count = 1):
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_query, "textDecorations": True, "answerCount" : n_count, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    search_results = "\n".join([text['snippet'] for text in search_results['webPages']['value']])

    return search_results

search_results = get_search_results(search_query)

def run_conversation(user_query):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": search_query}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_search_results",
                "description": "Get the search results from internet. Use this function when query is about latest events",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The query to search",
                        },
                        "n_count": {"type": "string",
                                    "description" : "The count of search results"},
                    },
                    "required": ["search_query"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_search_results": get_search_results,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            function_response = json.dumps({"search_result" : function_response})
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
    else:
        return response_message
    
print(run_conversation(search_query))
