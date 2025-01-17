import pandas as pd
from llm_provider import LLMProvider

a=LLMProvider(provider='huggingface',prompt_template='Given the salary of {salary}, what is the square of the salary?',
              hf_model='mistralai/Mathstral-7B-v0.1')

df=pd.DataFrame({'salary':[2,3,4,5,6],'emp_name':['shrajan','rinith','pradeep','manju','ganesh']})

b=a.process_row_by_row(df)
print(b)
