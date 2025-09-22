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
from typing import Literal, Optional
from langchain_neo4j import Neo4jGraph

from neo4j import GraphDatabase
from helper_ollama import create_question_embedding
  
# From query/question to cypher query language (cql) TODO
def create_question_cql(question:str):
    cql_query = ""
    return cql_query

def vectorize_property(element: Literal["node", "relationship"],
                   type_name:str, property_name:str,
                   session: Optional[Session] = None, 
                   kg:Optional[Neo4jGraph] = None):
    
    if session:
        runner = session.run # though Neo4j native driver session
    else:
        runner = kg.query # throgh Neo4jGraph Langchain object
    
    if element == "node":
        cprint(f"\nGenerating embeddings for (n:{type_name}) on n.{property_name}", "green")
        records = list(runner(f"""
            MATCH (n:{type_name})
            WHERE n.{property_name} IS NOT NULL AND n.{property_name} <> ''
            AND n.embedding IS NULL
            RETURN n.uuid AS uuid, n.{property_name} AS txt
            """))
        for r in records:
            vec = ollama.embed(model="nomic-embed-text", input=r["txt"])["embeddings"][0]
            runner(
                f"""
                MATCH (n:{type_name} {{uuid: $uuid}})
                SET n.embedding = $vec
                """,
                {"uuid": r["uuid"], "vec": vec},
            )
            print(f"  text: {r['txt']}\n  vec: {vec[:3]}")
            
    elif element == "relationship":
        
        cprint(f"\nGenerating embeddings for [r:{type_name}] on r.{property_name}", "green")
        records = list(runner(f"""
            MATCH ()-[r:{type_name}]-()
            WHERE r.{property_name} IS NOT NULL AND r.{property_name} <> ''
            AND r.embedding IS NULL
            RETURN r.uuid AS uuid, r.{property_name} AS txt
            """))
        for r in records:
            vec = ollama.embed(model="nomic-embed-text", input=r["txt"])["embeddings"][0]
            runner(
                f"""
                MATCH ()-[r:{type_name} {{uuid: $uuid}}]-()
                SET r.embedding = $vec
                """,
                {"uuid": r["uuid"], "vec": vec},
            )
            print(f"  text: {r['txt']}\n  vec: {vec[:3]}")

# Custom Neo4j Vector Search (for nodes)
def neo4j_vector_search(element: Literal["node", "relationship"] = "node", 
                        question:str = "", 
                        index:str = "", 
                        session: Optional[Session] = None, 
                        kg:Optional[Neo4jGraph] = None):
    
    """Search for similar nodes / relationships using the Neo4j vector index"""
      
    top_k = 10
    
    if element == "node":
        vector_search_query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $question_embedding) 
        YIELD node, score
        {create_question_cql(question)}
        RETURN score, node {{.* , embedding: ""}}
        """
    elif element == "relationship":
        vector_search_query = f"""
        CALL db.index.vector.queryRelationships($index_name, $top_k, $question_embedding) 
        YIELD relationship, score
        {create_question_cql(question)}
        RETURN score, relationship {{.* , embedding: ""}}
        ORDER BY score DESC
        """
        
    if session:
        runner = session.run # though Neo4j native driver session
    else:
        runner = kg.query # throgh Neo4jGraph Langchain object
        
    res = runner(vector_search_query, 
            {
            "index_name": index, 
            "top_k": top_k,
            "question_embedding": create_question_embedding(question)
            })
    
    result = list(res)
      
    return result

