from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print


@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text."""
    return len(text)


tools = {
    "get_text_length": get_text_length
}

llm = ChatMistralAI(
    model_name="mistral-small-2506"
)

llm_with_tool = llm.bind_tools([get_text_length])

messages = []

prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

# First LLM call
result = llm_with_tool.invoke(messages)
messages.append(result)

# Execute tool if requested
if result.tool_calls:
    tool_call = result.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_result = tools[tool_name].invoke(tool_args)

    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # Final LLM response
    final = llm_with_tool.invoke(messages)
    print(final.content)

else:
    print(result.content)