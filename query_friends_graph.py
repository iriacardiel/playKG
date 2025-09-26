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
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain_neo4j import GraphCypherQAChain

# === Local helpers
from helpers import helper_neo4j
import prompts

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

load_dotenv()  # Load local environment variables

URI = "bolt://localhost:" + os.environ.get("URI_PORT")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE")
MODEL_NAME = os.getenv("MODEL_NAME")

# Connect
kg = Neo4jGraph(url=URI, username=NEO4J_USER, password=NEO4J_PWD, database=NEO4J_DB)
cprint(f"\nConnected to Neo4j database: {NEO4J_DB}", "green")

# Initialize the chat model
llm = ChatVertexAI(
  model=MODEL_NAME,   # or "gemini-2.5-pro"
  temperature=0.2,
  max_output_tokens=1000,
  location="us-central1",     # or "europe-west1"
)

cprint(f"\nUsing LLM {MODEL_NAME} through VertexAI API.", "green")

# ----------------------------------------------------
# Vector Search with predefined queries:
#  user query -> predefined CQL query -> context -> LLM -> answer
# ----------------------------------------------------
def vector_search_QA(
    query: str,
    index: str, 
    source_property : str,
    main_property : str,
    top_k: int) -> str:

    # Query KG 
    # --------
    result = helper_neo4j.neo4j_KGRAG_search(
                                runner = kg.query,
                                query = query, 
                                index = index,
                                source_property = source_property,
                                main_property =main_property,
                                top_k = top_k
                                )
    
    # Build prompt
    # ------------
    llm_context = result.get("combined_context", "")
    prompt = prompts.get_prompt("system").format(query=query, llm_context=llm_context)
    cprint(prompt, "magenta")
    
    # Call LLM
    # --------
    llm_output = llm.invoke(prompt)
    response = llm_output.content
    cprint(response, "cyan")
    print("#"*60)
    print()
    return response, llm_context
      
  
# -----------------------------------------------------------------------------
# CQL Search with LLM Generated queries
#  user query -> LLM -> Generated CQL query -> context -> LLM -> answer
# -----------------------------------------------------------------------------
def generative_CQL_search_QA(query):

    # Prompt template for the LLM to generate the Cypher Query
    # --------------------------------------------------------
    
    prompt_cypher_genai = PromptTemplate(
        input_variables=["schema", "question"], 
        template=prompts.get_prompt("cypher_genai")
    )
    
  
    # Langchain Chain: user query -> LLM -> Generated CQL query -> context -> LLM -> answer
    # -------------------------------------------------------------------------------------
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=kg,
        verbose=True,
        cypher_prompt=prompt_cypher_genai,
        allow_dangerous_requests = True,
        return_intermediate_steps=True,  # needed to grab cypher + rows
        )

    cprint(query, "magenta")
    
    # Call LLM Chain
    # --------------
    out = cypher_chain.invoke(query)
    response = out.get("result","")
    steps = out.get("intermediate_steps") or []
    if steps:
      cypher = steps[0].get("query") 
      context = steps[1].get("context")
    cprint(response, "cyan")
    print("#"*60)
    print()

    return response, cypher, context

  
# KG RAG Search
if __name__ == "__main__":
  
  vector_search_QA(query = "Does any girl have short hair?", 
                   index = "person_node_idx", 
                   source_property = "text", 
                   main_property = "name", 
                   top_k = 3)
  
  vector_search_QA(query = "Which company has more employes belonging to the graph?", 
                   index = "company_node_idx", 
                   source_property = "text",  
                   main_property = "name", 
                   top_k = 3)
  
  vector_search_QA(query = "Who's Iria's best friend?", 
                   index = "know_relationship_idx", 
                   source_property = "text",  
                   main_property = "name", 
                   top_k = 3)

  generative_CQL_search_QA(query = "Who are Iria's friends?")
  generative_CQL_search_QA(query = "List people that work at Indra but do not know Paula?")
  generative_CQL_search_QA(query = "Who is closest to Cristina and by how much distance?")