from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage , ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# Creating tools

# Weather tools

@tool
def get_weather(city : str) -> str:
    """Get Current weather of a city"""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f"Weather in {city}: {desc}, {temp}°C"

# Tavily news tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city : str) -> str:
    """Get latest news about the city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

llm = ChatMistralAI(
    model_name="mistral-small-2506"
)

# tools = {
#     "get_weather" : get_weather,
#     "get_news" : get_news
# }

# llm_with_tool = llm.bind_tools([get_weather , get_news])

# # Agent Loop

# messages = []

# print("City Intelligence System")
# print("Type Exit to quit")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break
#     messages.append(HumanMessage(user_input))

#     while True:
#         result = llm_with_tool.invoke(messages)

#         messages.append(result)

#         #if tool is required

#         if result.tool_calls:
#             for tool_call in result.tool_calls:
#                 tool_name = tool_call['name']

#                 #Human in the loop
#                 confirm = input(f"Agent wants to call {tool_name} Approve (ys/no)")

#                 if confirm.lower() == 'no':
#                     print("tool call denied and I cannot get the latest information")
#                     break

#                 # execute tool
#                 tool_result = tools[tool_name].invoke(tool_call)

#                 messages.append(ToolMessage(
#                     content = tool_result,
#                     tool_call_id = tool_call["id"]
#                 ))

#             continue
#         else:
#             print("\n  Final Answer:\n")
#             print(result.content)
#             print('\n' + "="*50 + "\n")
#             break

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)

agent = create_agent(
    llm,
    tools = [get_weather , get_news],
    system_prompt="You are a helpfull city assistant",
    middleware = [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You :")
    if user_input.lower() == 'exit':
        break
    result = agent.invoke({
        "messages": [{"role" : "user" , "content": user_input}]
    })

    print("bot : " , result['messages'][-1].content)