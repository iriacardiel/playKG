"""
Ollama Operations Module
"""
import os
from dotenv import load_dotenv  

from termcolor import cprint

import ollama

load_dotenv()  # Load local environment variables
EMB_MODEL = os.getenv("EMB_MODEL", "nomic-embed-text")

# From user query/query to query embedding
def create_embedding(input_text:str):
    vec = ollama.embed(model=EMB_MODEL, input=input_text)["embeddings"][0] 
    return vec
  