from termcolor import cprint
import ollama

# From user query/query to query embedding
def create_embedding(input_text:str, verbose:bool=True):
    cprint(f"\nGenerating embeddings", "green")
    vec = ollama.embed(model="nomic-embed-text", input=input_text)["embeddings"][0] 
    print(f"  input text: '{input_text[0:50]}'...\n  emb vec: {vec[:10]}\n")
    return vec
  