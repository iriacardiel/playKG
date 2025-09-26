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
import prompts

# ------------------------ Setup ------------------------
load_dotenv()

URI = "bolt://localhost:" + os.environ.get("URI_PORT", "7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

from query_friends_graph import vector_search_QA, generative_CQL_search_QA

# ------------------------ Handlers ------------------------
def handle_vector(v_query, v_index_choice,
                  v_source_property, v_main_property, v_top_k):
    """Run neo4j_KGRAG_search and return (map_html, answer_text, cypher_text)."""
    llm_context = ""
    if not v_query.strip():
        return "",""
    try:
        ans, llm_context = vector_search_QA(
                                query=v_query,
                                index=v_index_choice,
                                source_property=v_source_property,
                                main_property=v_main_property,
                                top_k=v_top_k
                                )

    except Exception as e:
        # minimal error surface
        return f"Vector search error: {e}", ""
    
    return ans, llm_context

def handle_cql(cql_question):
    """Run GraphCypherQAChain and return (answer_text, cypher_text)."""
    q = (cql_question or "").strip()
    answer, cypher = "", ""
    if not q:
        return "", "", ""
    try:
        answer, cypher, llm_context = generative_CQL_search_QA(q)
    except Exception as e:
        return f"CQL error: {e}", ""

    return answer, cypher, str(llm_context)

# ------------------------ UI ------------------------
with gr.Blocks(css="") as demo:
    
    # Context through Vector search
    gr.Markdown("### Vector Search")
    v_query = gr.Textbox(label="Query", lines=2, placeholder="Does any girl have short hair?")
    with gr.Row():
        v_index_choice = gr.Dropdown(
            choices=["person_node_idx", "company_node_idx", "know_relationship_idx"],
            value="person_node_idx",
            label="index"
        )
    with gr.Row():
        v_source_property = gr.Textbox(label="source_property", value="text")
        v_main_property = gr.Textbox(label="main_property", value="name")
        v_top_k = gr.Number(label="top_k", value=5, precision=0)
        
    btn_vec = gr.Button("Run")
    v_ans = gr.Textbox(label="Answer", interactive=False, lines=3)
    v_context = gr.Textbox(label="Retrieved Context", interactive=False, lines=3)

    # Context through Generative CQL search
    gr.Markdown("### CQL GenAI")
    cql_question = gr.Textbox(label="Question", lines=2, placeholder="Who are Iria's friends?")
    btn_cql = gr.Button("Run")
    cql_ans = gr.Textbox(label="Answer", interactive=False, lines=3)
    cql_cypher = gr.Textbox(label="Generated Cypher", interactive=False, lines=3)
    cql_context = gr.Textbox(label="Retrieved Context", interactive=False, lines=3)

    # Wire up buttons
    btn_vec.click(
        handle_vector,
        inputs=[v_query, v_index_choice, v_source_property, v_main_property, v_top_k],
        outputs=[v_ans, v_context]
    )
    btn_cql.click(
        handle_cql,
        inputs=[cql_question],
        outputs=[cql_ans, cql_cypher, cql_context]
    )

demo.launch()
