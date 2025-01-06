"""
Contain class for generating prompt using given template and data
"""

class PromptHandler:
    """
    A utility class to handle prompt generation using a specified template.

    This class takes a prompt template during initialization and provides methods 
    to generate prompts from data rows (either as dictionaries or single values).

    Attributes:
        prompt_template (str): The template used to generate prompts.
    """
    
    def __init__(self, prompt_template: str):
        """
        Initialize the PromptHandler with a prompt template.

        Args:
            prompt_template (str): Template string with placeholders for formatting.
        """
        self.prompt_template = prompt_template

    def generate_prompt_df(self, **kwargs):
        """
        Generate a prompt from a dictionary-like input, typically a row from a DataFrame.

        Args:
            **kwargs: Key-value pairs that correspond to the placeholders in the prompt template.

        Returns:
            str: The generated prompt based on the input data.
        """
        return self.prompt_template.format(**kwargs)

    def generate_prompt_srs(self, args):
        """
        Generate a prompt from a single input, typically a value from a Series.

        Args:
            args : Input value to format into the prompt template.

        Returns:
            str: The generated prompt based on the input value.
        """
        return self.prompt_template.format(args)

    