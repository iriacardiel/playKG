from termcolor import cprint
import ollama

from dotenv import load_dotenv  
import os
load_dotenv()  # Load local environment variables

EMB_MODEL = os.getenv("EMB_MODEL", "nomic-embed-text")

# From user query/query to query embedding
def create_embedding(input_text:str):
    cprint(f"\nGenerating embeddings", "yellow")
    vec = ollama.embed(model="nomic-embed-text", input=input_text)["embeddings"][0] 
    print(f"  input text: '{input_text[0:50]}'...\n  emb vec: {vec[:10]}\n")
    return vec
  