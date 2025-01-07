"""
llm_provider module provides interface to interact with different LLM providers 
(Openai and Huuging face as of now). It includes utilities for handling prompts,
processing data row-by-row or in chunks, and managing token limits
"""

import openai
import pandas as pd
from transformers import pipeline
from .prompt_handler import PromptHandler

class LLMProvider:
    """
    Class to querry  LLM provided by openai and Hugging face.
      
    Attributes:
      provider (str) : Name of llm provider
      api_key(str) : Api key for accessing llm provider
      prompt_template(str) : Template to format prompts
      max_tokens (int) : maximum token limit for each querry
      hf_model (str) : hugging face model name, required if provider is hugging face default None
    
    """

    def __init__(self,provider:str,api_key:str,prompt_template:str,max_tokens,hf_model=None):
        
        """
        Initializes the LLMProvider class.

        Args:
            provider (str): Name of the LLM provider ('openai' or 'huggingface').
            api_key (str): API key for accessing the LLM provider.
            prompt_template (str): Template to format prompts.
            max_tokens (int): Maximum token limit for each query.
            hf_model (str, optional): Hugging Face model name, required if provider is 'huggingface'.
        """

        self.provider=provider
        self.api_key=api_key
        self.max_tokens=max_tokens
        self.hf_model=hf_model
        self.prompt_handler=PromptHandler(prompt_template)
        
        #initialise huggingface pipline if provider is huggingface
        if self.provider=="huggingface" and self.hf_model:
             self.hf_pipline=pipeline('text-generation',model=hf_model)
        
    def query(self, prompt: str):
        """
        Query the respective LLM provider (OpenAI or Hugging Face).

        Args:
            prompt (str): Input prompt to query the LLM.

        Returns:
            str: Response generated by the LLM.
        
        """

        if self.provider == "openai":
            return self._query_openai(prompt)
        elif self.provider == "huggingface":
            return self._query_huggingface(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _query_openai(self, prompt: str):
        """
        Query OpenAI for a response 
         Args:
            prompt (str): Input prompt to query the OpenAI LLM.

        Returns:
            str: Response generated by OpenAI.
            """
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
        Args:
            prompt (str): Input prompt to query the OpenAI LLM.

        Returns:
            str: Response generated by OpenAI.
            """
        if not self.hf_pipeline:
            raise ValueError("Hugging Face pipeline is not initialized. Please provide a valid Hugging Face model.")
        response = self.hf_pipeline(prompt, max_length=self.max_tokens, truncation=True)
        return response[0]["generated_text"]

    def _convert_to_dicts(self,rows):
        """If input is Pandas Data frame converts it to dictionary,
           If Pandas Series converts it to list.
           args:
            rows (pd.DataFrame or pd.Series): Input data.

           Returns:
            list: List of dictionaries or list of values.
        """
        #if data frame converts it to dictionary
        if isinstance(rows,pd.DataFrame):
            return rows.to_dict(orient='records')
        elif isinstance(rows,pd.Series): #if series converts to list
             return list(rows)
        else:
             raise TypeError("Pass either pandas Data Frame or pandas Series")
    
    def process_row_by_row(self, rows):
            """
            Process input row-by-row. This method generates a prompt for each row 
            and queries the LLM for a response.

            Args :
                 rows (pandas series or dataframe) : Input data

            Out put :
                    List : List of Response generated by for each row
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
            """
        Process input data in one big chunk. Combines all rows into a single prompt
        and queries the LLM.

        Args:
            rows (pd.DataFrame or pd.Series): Input data.

        Returns:
            str: Response generated by the LLM for the combined prompt.
            """
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
    
    def process_in_chunks(self,rows):
        """
        Process input data in chunks. Splits rows into chunks that fit within the token limit
        and queries the LLM for each chunk.

        Args:
            rows (pd.DataFrame or pd.Series): Input data.

        Returns:
            list: List of responses for each chunk.
        """
        

        def make_chunks(prompts,max_tokens):
            """
            Split prompts into chunks based on the maximum token limit.

            Args:
                prompts (list): List of prompt strings.
                max_tokens (int): Maximum token limit for each chunk.

            Returns:
                list: List of chunks.
            """
            chunks=[]
            sub_chunks=[]
            token=0

            for prompt in prompts:
                prompt_len = self.count_tokens(prompt)
                if prompt_len+token<=max_tokens:
                            sub_chunks.append(prompt)
                            token+=prompt_len
                else: 
                    chunks.append("\n".join(sub_chunks))
                    sub_chunks=[prompt]
                    token=prompt_len
            
            #if any sub chunks are left over
            if sub_chunks:
                chunks.append("\n".join(sub_chunks))
        
        if isinstance(rows,pd.DataFrame):
            rows=self._convert_to_dicts(rows)
            prompts=[self.prompt_handler.generate_prompt_df(**row) for row in rows]
            chunks=make_chunks(prompts,self.max_tokens)
            results=[]
            for chunk in chunks:
                 result=self.query(chunk)
                 result.append(result)
            return results
        
        elif isinstance(rows,pd.Series):
            rows= self._convert_to_dicts(rows)
            prompts=[self.prompt_handler.generate_prompt_srs(row) for row in rows]
            chunks=make_chunks(prompts)
            results=[]
            for chunk in chunks:
                 result=self.query(chunk)
                 results.append(result)
            return results
        
        else:
             raise TypeError("Input should be of type Pandas DataFrame or Pandas Series")
                          
    def count_tokens(self,text:str):
        """
        Count the number of tokens in the given text.

        Args:
            text (str): Input text.

        Returns:
            int: Number of tokens in the text.
        """
        return len(text.split())

