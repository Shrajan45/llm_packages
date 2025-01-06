import openai
import pandas as pd
from transformers import pipeline
from .prompt_handler import PromptHandler


class LLMProvider():
    def __init__(self,provider:str,api_key:str,prompt_template:str,max_tokens,hf_model=None):
        self.provider=provider
        self.api_key=api_key
        self.max_tokens=max_tokens
        self.hf_model=hf_model
        self.prompt_handler=PromptHandler(prompt_template)

        if self.provider=="huggingface" and self.hf_model:
             self.hf_pipline=pipeline('text-generation',model=hf_model)
        
    def query(self, prompt: str):
        """
        Query the respective LLM provider (OpenAI or Hugging Face).
        """
        if self.provider == "openai":
            return self._query_openai(prompt)
        elif self.provider == "huggingface":
            return self._query_huggingface(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _query_openai(self, prompt: str):
        """
        Query OpenAI for a response """
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        return response.choices[0].text.strip()

    def _query_huggingface(self, prompt: str):
        """
        Query Hugging Face for a response.
        """
        if not self.hf_pipeline:
            raise ValueError("Hugging Face pipeline is not initialized. Please provide a valid Hugging Face model.")
        response = self.hf_pipeline(prompt, max_length=self.max_tokens, truncation=True)
        return response[0]["generated_text"]

    def _convert_to_dicts(self,rows):
        if isinstance(rows,pd.DataFrame):
            return rows.to_dict(orient='records')
        elif isinstance(rows,pd.Series):
             return list(rows)
        else:
             raise TypeError("Pass either pandas Data Frame or pandas Series")
    
    def process_row_by_row(self, rows):
            """
            Process input row-by-row. This method generates a prompt for each row 
            and queries the LLM for a response.
            """

            if isinstance(rows,pd.DataFrame):
                    rows = self._convert_to_dicts(rows)
                    results = []
                    for row in rows:
                        prompt = self.prompt_handler.generate_prompt_df(**row)  # Use prompt_handler to generate the prompt
                        results.append(self.query(prompt))  # Query with the generated prompt
                    return results
            elif isinstance(rows,pd.Series):
                    rows = self._convert_to_dicts(rows)
                    results = []
                    for row in rows:
                        prompt = self.prompt_handler.generate_prompt_srs(row)  # Use prompt_handler to generate the prompt
                        results.append(self.query(prompt))  # Query with the generated prompt
                    return results
            else:
                 raise TypeError("Input should be of type Pandas DataFrame or Series")
    
    def process_in_one_big_chunk(self,rows):
            if isinstance(rows,pd.DataFrame):          
                    rows=self._convert_to_dicts(rows)
                    formated_rows=[self.prompt_handler.generate_prompt_df(**row) for row in rows]
                    prompts="\n".join(formated_rows)
                    if self.count_tokens(prompts)>self.max_tokens:
                        raise ValueError("Total prompt exceeds the maximum token limit")            
                    return self.querry(prompts)
            elif isinstance(rows,pd.Series):
                    rows=self._convert_to_dicts(rows)
                    formated_rows=[self.prompt_handler.generate_prompt_srs(row) for row in rows]
                    prompts="\n".join(formated_rows)
                    if self.count_tokens(prompts)>self.max_tokens:
                        raise ValueError("Total prompt exceeds the maximum token limit")            
                    return self.querry(prompts)             
            else:
                 raise TypeError("Input should be of type Pandas DataFrame or Series")
    def count_tokens(self,text:str):
         return len(text.split())
