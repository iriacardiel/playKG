"""
Friends KG - Search & Retrieval Script
"""
# === Dependencies
import os
from dotenv import load_dotenv
from termcolor import cprint
from pprint import pprint


# === Neo4j / LangChain
from langchain_neo4j import Neo4jGraph

# === Local helpers
from helpers import helper_neo4j

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

load_dotenv()  # Load local environment variables

URI = "bolt://localhost:" + os.environ.get("URI_PORT")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE")

# -----------------------------------------------------------------------------
# Vector Search
# -----------------------------------------------------------------------------

# Connect
kg = Neo4jGraph(url=URI, username=NEO4J_USER, password=NEO4J_PWD, database=NEO4J_DB)
cprint(f"\nConnected to Neo4j database: {NEO4J_DB}", "green")

# KG RAG Search
if __name__ == "__main__":

    # Query Person Nodes
    result = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                element = "node",
                                query = "Who likes horror movies?", 
                                index = "person_node_idx",
                                source_property = "text",
                                main_property = "name",
                                top_k = 1
                                )
    pprint(result, width = 200, sort_dicts=False, indent=2)

    file = "data/friends/friends_context_1.txt"
    with open(file, 'w', encoding='utf-8') as f:
      f.write(result.get("combined_context", ""))

    # Query Company Nodes
    result  = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                    element = "node",
                                    query = "Which company works for the army?",
                                    index = "company_node_idx",
                                    source_property = "text",
                                    main_property = "name",
                                    top_k = 1
                                  )
    pprint(result, width = 200, sort_dicts=False, indent=2)

    file = "data/friends/friends_context_2.txt"
    with open(file, 'w', encoding='utf-8') as f:
      f.write(result.get("combined_context", ""))
      
      
    # Query KNOWS Relationships
    result  = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                  element = "relationship",
                                  query = "Who's Daniel's girlfriend?",
                                  index = "know_relationship_idx",
                                  source_property = "text",
                                  main_property = "name",
                                  top_k = 1
                                  )
    pprint(result, width = 200, sort_dicts=False)

    file = "data/friends/friends_context_3.txt"
    with open(file, 'w', encoding='utf-8') as f:
      f.write(result.get("combined_context", ""))
