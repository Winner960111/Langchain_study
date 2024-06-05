# from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
# # for defining class
# from langchain_core.pydantic_v1 import BaseModel
# from langchain_experimental.tabular_synthetic_data.openai import (OPENAI_TEMPLATE, create_openai_data_generator,)
# from langchain_experimental.tabular_synthetic_data.prompts import (
#     SYNTHETIC_FEW_SHOT_PREFIX,
#     SYNTHETIC_FEW_SHOT_SUFFIX,
# )
# from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# class MedicalBilling(BaseModel):
#     patient_id: int
#     patient_name: str
#     diagnosis_code: str
#     procedure_code: str
#     total_charge: float
#     insurance_claim_amount: float

# examples = [
#     {
#          "example": """Patient ID: 123456, Patient Name: John Doe, Diagnosis Code: 
#         J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"""
#     },
#     {
#         "example": """Patient ID: 789012, Patient Name: Johnson Smith, Diagnosis 
#         Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"""
#     },
#     {
#         "example": """Patient ID: 345678, Patient Name: Emily Stone, Diagnosis Code: 
#         E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"""
#     },
# ]

# OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

# prompt_template = FewShotPromptTemplate(
#     prefix=SYNTHETIC_FEW_SHOT_PREFIX,
#     examples=examples,
#     suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
#     input_variables=["subject", "extra"],
#     example_prompt=OPENAI_TEMPLATE
# )

# synthetic_data_generator = create_openai_data_generator(
#     output_schema=MedicalBilling,
#     llm=ChatOpenAI(
#         temperature=1,
#         api_key=api_key
#     ),
#     prompt=prompt_template
# )

# # generate sythetic Data

# synthetic_results = synthetic_data_generator.generate(
#     subject="medical_billing",
#     extra = "the name must be chosen at random. Make it something you wouldn't normally choose.",
#     runs=10
# )

# print(synthetic_results)



## other implementations

from langchain_experimental.synthetic_data import (
    DatasetGenerator,
    create_data_generation_chain,
)
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)
chain = create_data_generation_chain(model)

# print(chain({"fields": ["blue", "yellow"], "preferences":{}}))
# print(chain(
#     {
#         "fields": {"colors": ["blue", "yellow"]},
#         "preferences": {"style": "Make it in a style of a weather forecast."},
#     }
# ))
# print(chain(
#     {
#         "fields": {"actor": "Tom Hanks", "movies": ["Forrest Gump", "Green Mile"]},
#         "preferences": None,
#     }
# ))
print(chain(
    {
        "fields": [
            {"actor": "Tom Hanks", "movies": ["Forrest Gump", "Green Mile"]},
            {"actor": "Mads Mikkelsen", "movies": ["Hannibal", "Another round"]},
        ],
        "preferences": {"minimum_length": 200, "style": "gossip"},
    }
))