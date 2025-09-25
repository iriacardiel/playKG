# Dependencies

import os
from dotenv import load_dotenv  
import yaml
from pathlib import Path
from pprint import pprint
from termcolor import cprint
from langchain_neo4j import Neo4jGraph
import json

from helpers import helper_folium, helper_leaflet, helper_neo4j, helper_ollama

# Environment variables

load_dotenv()  # Load local environment variables

URI = "bolt://localhost:" + os.environ.get("URI_PORT")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PWD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")    # 👈 choose DB here

cprint(f"Connecting to Neo4j at {URI} with user {NEO4J_USER} and password {NEO4J_PWD}", "blue")

# Load cypher queries

queries = yaml.safe_load(Path("data/friends/queries_friends.yaml").read_text())

# Neo4j Langchain wrapper instance

kg = Neo4jGraph(url=URI, username=NEO4J_USER, password=NEO4J_PWD, database=NEO4J_DB)

# KG RAG Search

# Query Nodes
result = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                             element = "node",
                             query = "Who dances?", 
                             index = "person_node_idx",
                             source_property = "text",
                             main_property = "name",
                             top_k = 5
                             )
pprint(result, width = 200, sort_dicts=False, indent=2)
file = "data/friends/friends_context_1.txt"
with open(file, 'w', encoding='utf-8') as f:
  f.write(result.get("combined_context", ""))

result  = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                element = "node",
                                query = "Which company investigates Cancer?",
                                index = "company_node_idx",
                                source_property = "text",
                                main_property = "name",
                                top_k = 5
                              )
pprint(result, width = 200, sort_dicts=False, indent=2)
file = "data/friends/friends_context_2.txt"
with open(file, 'w', encoding='utf-8') as f:
  f.write(result.get("combined_context", ""))
  
  
# Query Relationships
result  = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                              element = "relationship",
                              query = "Who is helping Iria at work?",
                              index = "know_relationship_idx",
                              source_property = "text",
                              main_property = "name",
                              top_k = 5
                              )
pprint(result, width = 200, sort_dicts=False)

file = "data/friends/friends_context_3.txt"
with open(file, 'w', encoding='utf-8') as f:
  f.write(result.get("combined_context", ""))
