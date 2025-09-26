"""
Friends KG — Minimal Gradio UI
Two inputs + one map. Minimal controls, with send buttons.
Run: python app_gradio.py
"""
import os, json
import gradio as gr
import folium
from dotenv import load_dotenv
from termcolor import cprint
from pprint import pprint

# Neo4j / LangChain
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate

# Local helper
from helpers import helper_neo4j

# ------------------------ Setup ------------------------
load_dotenv()

URI = "bolt://localhost:" + os.environ.get("URI_PORT", "7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

kg = Neo4jGraph(url=URI, username=NEO4J_USER, password=NEO4J_PWD, database=NEO4J_DB)
cprint(f"Connected to Neo4j database: {NEO4J_DB}", "green")

llm = ChatVertexAI(
    model=MODEL_NAME,
    temperature=0.2,
    max_output_tokens=1000,
    location=LOCATION,
)
cprint(f"Using LLM {MODEL_NAME} through VertexAI API.", "green")

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

prompt1 = PromptTemplate(
    input_variables=["schema", "question"],
    template=PROMPT_TEMPLATE_CYPHER_GENAI
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=kg,
    verbose=True,
    cypher_prompt=prompt1,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,  # needed to grab cypher + rows
)

def prettyGenAICypherChain(question: str) -> str:
      cprint(question, "magenta")
      out = cypher_chain.invoke(question)
      response = out.get("result","")
      steps = out.get("intermediate_steps") or []
      if steps:
        cypher = steps[0].get("query") 
        context = steps[1].get("context")
      cprint(response, "cyan")
      print("#"*60)
      print()
      return response, cypher, context


PROMPT_TEMPLATE = """
    Anser to the query based on the following context retrieved from a Knowledge Graph through Vector Cosine Similarity.
        
    Query: `{query}`
    Context:\n`{llm_context}`
    """

def prettyVectorSearchChain(query:str, llm_context:str) -> str:
      prompt2 = PROMPT_TEMPLATE.format(query=query, llm_context=llm_context)
      cprint(prompt2, "magenta")
      llm_output = llm.invoke(prompt2)
      response = llm_output.content
      cprint(response, "cyan")
      print("#"*60)
      print()
      return response

# ------------------------ Handlers ------------------------
def handle_vector(v_query, v_element, v_index_choice, v_index_custom,
                  v_source_property, v_main_property, v_top_k):
    """Run neo4j_KGRAG_search and return (map_html, answer_text, cypher_text)."""
    index = v_index_custom.strip() if v_index_choice == "custom" else v_index_choice
    llm_context = ""
    if not v_query.strip():
        return "","", ""
    try:
        result = helper_neo4j.neo4j_KGRAG_search(
            runner=kg.query,
            element=v_element,
            query=v_query.strip(),
            index=index,
            source_property=v_source_property.strip(),
            main_property=v_main_property.strip(),
            top_k=int(v_top_k),
        )
        pprint(result, width = 200, sort_dicts=False, indent=2)
        llm_context = result.get("combined_context", "")
        with open("data/friends/friends_context_2.txt", 'w', encoding='utf-8') as f:
            f.write(llm_context)        
        # Call LLM
        # --------
        ans = prettyVectorSearchChain(v_query, llm_context)


    except Exception as e:
        # minimal error surface
        return f"Vector search error: {e}", "", ""
    
    return ans, llm_context, ""

def handle_cql(cql_question):
    """Run GraphCypherQAChain and return (answer_text, cypher_text)."""
    q = (cql_question or "").strip()
    answer, cypher = "", ""
    if not q:
        return "", "", ""
    try:
        answer, cypher, context_rows = prettyGenAICypherChain(q)
    except Exception as e:
        return f"CQL error: {e}", ""

    return answer, cypher, str(context_rows)

# ------------------------ UI ------------------------
with gr.Blocks(css="") as demo:
    # Vector search controls (query box + selectors)
    gr.Markdown("### Vector Search")
    v_query = gr.Textbox(label="Query", lines=2, placeholder="Does any girl have short hair?")
    with gr.Row():
        v_element = gr.Dropdown(choices=["node", "relationship"], value="node", label="element")
        v_index_choice = gr.Dropdown(
            choices=["person_node_idx", "company_node_idx", "know_relationship_idx", "custom"],
            value="person_node_idx",
            label="index"
        )
        v_index_custom = gr.Textbox(label="custom index (if selected)", placeholder="my_index_name")
    with gr.Row():
        v_source_property = gr.Textbox(label="source_property", value="text")
        v_main_property = gr.Textbox(label="main_property", value="name")
        v_top_k = gr.Number(label="top_k", value=5, precision=0)
    btn_vec = gr.Button("Run vector search")

    gr.Markdown("### CQL GenAI")
    cql_question = gr.Textbox(label="Question", lines=2, placeholder="Who are Iria's friends?")
    btn_cql = gr.Button("Run CQL")

    # Outputs (minimal)
    ans_out = gr.Textbox(label="Answer", interactive=False, lines=3)
    cypher_out = gr.Textbox(label="Generated Cypher", interactive=False, lines=3)
    context_out = gr.Textbox(label="Retrieved Context", interactive=False, lines=3)

    # Wire up buttons
    btn_vec.click(
        handle_vector,
        inputs=[v_query, v_element, v_index_choice, v_index_custom, v_source_property, v_main_property, v_top_k],
        outputs=[ans_out, context_out, cypher_out]
    )
    btn_cql.click(
        handle_cql,
        inputs=[cql_question],
        outputs=[ans_out, cypher_out, context_out]
    )

demo.launch()
