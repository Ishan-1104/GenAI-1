from langchain.tools import tool

@tool  #decorator for creating tool 
def get_greeting(name : str) -> str :
    """Generate a greeting message for user"""  #docstring

    return f"Hello {name}, Welcome to teh AI world"

result = get_greeting.invoke({"name" : "Ishan"})
print(result)
print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)