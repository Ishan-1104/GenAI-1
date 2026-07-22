from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model_name="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

# Step by Step Manual Flow

# # Format the prompt
# formatted_prompt = prompt.format_messages(topic="Machine Learning")

# # Call the model Manually
# response = model.invoke(formatted_prompt)

# # Parse the output manually
# final_output = parser.parse(response.content)

chain = prompt | model | parser

result = chain.invoke("Machine Learning")

print(result)