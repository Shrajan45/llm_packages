class PromptHandler:

    def __init__(self,prompt_template:str):
        self.prompt_template=prompt_template

    def generate_prompt_df(self,**kwargs):
        return self.prompt_template.format(**kwargs)
    
    def generate_prompt_srs(self,args):
        return self.prompt_template.format(args)
    