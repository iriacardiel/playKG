from neo4j._sync.work.session import Session
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv  
import yaml
from pathlib import Path
from pprint import pprint
from termcolor import cprint
import ollama
import requests
from typing import Literal

from neo4j import GraphDatabase

# From user query/question to question embedding
def create_question_embedding(question:str):
    cprint(f"\nGenerating embeddings for question '{question}'", "green")
    vec = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0] 
    print(f"  text: {question}\n  vec: {vec[:10]}\n")
    return vec
  