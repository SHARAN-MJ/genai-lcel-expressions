## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
LangChain Expression Language (LCEL) simplifies interactions with large language models (LLMs) by creating reusable and structured expressions. This task involves:

1. Designing an LCEL expression with dynamic prompt parameters (e.g., topic and length).
2. Using three essential components: Prompt- A structured input with placeholders for parameters, Model- An LLM used to process the prompt and Output Parser- A parser to interpret the model's output.
3. Demonstrating the LCEL expression's functionality in generating structured, relevant outputs.

### DESIGN STEPS:

#### STEP 1: Define the Parameters
Identify the parameters (topic and length) to allow dynamic customization of prompts.

#### STEP 2: Design the Prompt Template
Create a structured prompt template with placeholders for parameters.

#### STEP 3: Select the Model
Use an LLM, such as OpenAI's GPT, to process the prompt.

#### STEP 4: Implement the Output Parser
Design an output parser to format and structure the model's output.

#### STEP 5: Integrate Components into an LCEL Expression
Combine the prompt template, model, and output parser into a LangChain pipeline.

#### STEP 6: Evaluate with Examples
Test the LCEL expression using multiple input values for topic and length.

### PROGRAM:
NAME: SHARAN MJ
REG NO: 212222240097

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.schema.output_parser import StrOutputParser

# Define the PromptTemplate
prompt = PromptTemplate(
    template="""
You are a travel assistant. Based on the following inputs, recommend a destination:
- Preferred activity: {activity}
- Budget (in USD): {budget}

Provide a response strictly in JSON format:
{{
    "destination": "<destination>",
    "activity": "<activity>",
    "cost": "<cost>"
}}
""",
    input_variables=["activity", "budget"],
)

# Define the Output Parser
response_schemas = [
    ResponseSchema(name="destination", description="Recommended travel destination"),
    ResponseSchema(name="activity", description="Suggested activity at the destination"),
    ResponseSchema(name="cost", description="Estimated cost in USD for the trip"),
]
output_parser = StrOutputParser()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-0613", temperature=0)

# Create the LangChain Expression (LLM Chain)
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)

# Test the chain with an example
input_data = {"activity": "hiking", "budget": 1000}
result = chain.run(input_data)

# Parse the structured output
parsed_result = output_parser.parse(result)

# Display the result
print("Recommendation:", parsed_result)
```

### OUTPUT:

![Screenshot 2025-04-19 103955](https://github.com/user-attachments/assets/2b8d20a6-c572-4cc9-9810-f46c45232d55)


### RESULT:
  Thus, the LangChain Expression Language (LCEL) expression that utilizes two prompt parameters and three key components (prompt, model, and output parser) was designed and implemented successfully. And also evaluated its functionality by analyzing relevant examples of its application in real-world scenarios.
