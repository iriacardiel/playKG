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
def RAG_search_QA_examples():
  
    PROMPT_TEMPLATE = """
    Anser to the query based on the following context retrieved from a Knowledge Graph through Vector Cosine Similarity.
        
    Query: `{query}`
    Context:\n`{llm_context}`
    """

    def prettyVectorSearchChain(query:str, llm_context:str) -> str:
      prompt = PROMPT_TEMPLATE.format(query=query, llm_context=llm_context)
      cprint(prompt, "magenta")
      llm_output = llm.invoke(prompt)
      response = llm_output.content
      cprint(response, "cyan")
      cprint("#"*60)
  
    # Query Person Nodes to fetch context
    # ------------------------------------
    query = "Does any girl have short hair?"
    result = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                element = "node",
                                query = query, 
                                index = "person_node_idx",
                                source_property = "text",
                                main_property = "name",
                                top_k = 5
                                )
    #pprint(result, width = 200, sort_dicts=False, indent=2)
    llm_context = result.get("combined_context", "")
    with open("data/friends/friends_context_3.txt", 'w', encoding='utf-8') as f:
      f.write(llm_context)
    
    # Call LLM
    # --------
    prettyVectorSearchChain(query, llm_context)
      

    # Query Company Nodes to fetch context
    # ------------------------------------
    query = "Which company has more employes belonging to the graph?"
    result  = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                    element = "node",
                                    query = query,
                                    index = "company_node_idx",
                                    source_property = "text",
                                    main_property = "name",
                                    top_k = 5
                                  )
    #pprint(result, width = 200, sort_dicts=False, indent=2)
    llm_context = result.get("combined_context", "")
    with open("data/friends/friends_context_2.txt", 'w', encoding='utf-8') as f:
      f.write(llm_context)
      
    # Call LLM
    # --------
    prettyVectorSearchChain(query, llm_context)
      
      
    # Query KNOWS Relationships to fetch context
    # -----------------------------------------
    query = "Who's Iria's best friend?"
    result  = helper_neo4j.neo4j_KGRAG_search(runner = kg.query,
                                  element = "relationship",
                                  query = query,
                                  index = "know_relationship_idx",
                                  source_property = "text",
                                  main_property = "name",
                                  top_k = 5
                                  )
    #pprint(result, width = 200, sort_dicts=False)
    llm_context = result.get("combined_context", "")
    with open("data/friends/friends_context_3.txt", 'w', encoding='utf-8') as f:
      f.write(llm_context)
      
    # Call LLM
    # --------
    prettyVectorSearchChain(query, llm_context)
    
    
  
# -----------------------------------------------------------------------------
# CQL Search with LLM Generated queries
#  user query -> LLM -> Generated CQL query -> context -> LLM -> answer
# -----------------------------------------------------------------------------
def CQL_search_QA_examples():

  # Prompt template for the LLM to generate the Cypher Query
  # --------------------------------------------------------
  PROMPT_TEMPLATE_CYPHER_GENAI = """Task:Generate Cypher statement to query a graph database.
  Instructions:
  Use only the provided relationship types and properties in the schema.
  Do not use any other relationship types or properties that are not provided.
  Schema:
  {schema}
  Note: Do not include any explanations or apologies in your responses.
  Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
  Do not include any text except the generated Cypher statement.
  Examples: Here are a few examples of generated Cypher statements for particular questions:

  # Show people
  MATCH (p:Person) 
  RETURN p.name as name, p.age as age, p.gender, p.education as education
  ORDER BY age
  
  # Show companies
  MATCH (c:Company) 
  RETURN c.name as name, c.industry as industry
  ORDER BY industry
  
  # Show text descriptions of the different nodes
  MATCH (n) 
  RETURN labels(n), n.name, n.text
  
  # Show distances between people and Iria
  MATCH (p:Person {{name:"Iria"}}), (other:Person)
  WITH p, other, point.distance(p.location, other.location) AS distance_m
  RETURN p.name, other.name, round(distance_m/1000, 2) + " km" AS distance_km

  The question is:
  {question}"""


  prompt = PromptTemplate(
      input_variables=["schema", "question"], 
      template=PROMPT_TEMPLATE_CYPHER_GENAI
  )
  
  #helper_neo4j.show_schema(kg)
  
  prompt.pretty_print()
  
  # Langchain Chain: user query -> LLM -> Generated CQL query -> context -> LLM -> answer
  # -------------------------------------------------------------------------------------
  cypherChain = GraphCypherQAChain.from_llm(
      llm=llm,
      graph=kg,
      verbose=True,
      cypher_prompt=prompt,
      allow_dangerous_requests = True
      )

  def prettyCypherChain(question: str) -> str:
      response = cypherChain.invoke(question)
      cprint(response, "cyan")
      
  # Call LLM Chain
  # --------------
  prettyCypherChain("Who are Iria's friends?")
  prettyCypherChain("Which people work at Indra but are not friends with Javier? Do not consider that Javier can know himself obiously.")
  prettyCypherChain("Who is closest to Cristina? Who is furthest? By how much?")

  
# KG RAG Search
if __name__ == "__main__":
  RAG_search_QA_examples()
  CQL_search_QA_examples()
   
