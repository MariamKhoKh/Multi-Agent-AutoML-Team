import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

class Config:
    
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    @classmethod
    def validate(cls):
        required = [
            cls.AZURE_OPENAI_API_KEY,
            cls.AZURE_OPENAI_ENDPOINT,
            cls.AZURE_OPENAI_API_VERSION,
            cls.AZURE_OPENAI_DEPLOYMENT
        ]
        if not all(required):
            raise ValueError("Missing Azure OpenAI configuration in .env file")
    
    @classmethod
    def get_client(cls):
        cls.validate()
        return AzureOpenAI(
            api_key=cls.AZURE_OPENAI_API_KEY,
            api_version=cls.AZURE_OPENAI_API_VERSION,
            azure_endpoint=cls.AZURE_OPENAI_ENDPOINT
        )
