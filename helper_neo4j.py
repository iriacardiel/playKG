"""
Neo4j Vector Operations Module
"""

from typing import Literal, Optional, List, Dict, Any, Union, Callable, Mapping, Iterable
from neo4j._sync.work.session import Session
from langchain_neo4j import Neo4jGraph
from termcolor import cprint
  
from helper_ollama import create_embedding

def vectorize_property(
    runner: Callable[[str, Optional[Mapping[str, Any]]], Iterable],
    element: Literal["node", "relationship"]="node",
    node_label: Optional[str]="",
    rel_type: Optional[str]="",
    source_property: str="",
) -> None:
    """
    Generate and store vector embeddings for specified property of nodes or relationships.
    
    This function processes all nodes or relationships of a given label or type that have
    non-empty values for the specified property and generates embeddings for them.
    Only processes items that don't already have embeddings.
    
    Args:
        runner: Query execution function (session.run or kg.query)
        element: Type of graph element - either "node" or "relationship"
        node_label: Label of the node to process
        rel_type: Type of the relationship to process
        source_property: Name of the property to vectorize
        
    Note:
        Only one of 'node_label' or 'rel_type' must be provided.
    """
    
    # Input control
    
    if node_label and rel_type:
        raise ValueError("Only one of 'node_label' or 'rel_type' must be provided.")
    elif element == "node" and not node_label:
        raise ValueError("Provide 'node_label' for element 'node'.")
    elif element == "relationship" and not rel_type:
        raise ValueError("Provide 'rel_type' for element 'relationship'.")

    # Vectorize node property
    if element == "node":

        cprint(f"\nGenerating embeddings for (n:{node_label}) on n.{source_property}", "green")
        
        # Query for nodes without embeddings
        query = f"""
            MATCH (n:{node_label})
            WHERE 
                n.{source_property} IS NOT NULL AND 
                n.{source_property} <> '' AND 
                n.embedding IS NULL
            RETURN 
                n.uuid AS uuid,
                n.{source_property} AS txt
        """
        
        records = list(runner(query))
        
        # Generate embeddings for each record
        for record in records:
            vec = create_embedding(input_text=record["txt"])
            
            # Update node with embedding
            update_query = f"""
                MATCH (n:{node_label} {{uuid: $uuid}})
                SET n.embedding = $vec
            """
            runner(update_query, {"uuid": record["uuid"], "vec": vec})

    # Vectorize relationship property 
    elif element == "relationship":

        cprint(f"\nGenerating embeddings for [r:{rel_type}] on r.{source_property}", "green")
        
        # Query for relationships without embeddings
        query = f"""
            MATCH ()-[r:{rel_type}]-()
            WHERE 
                r.{source_property} IS NOT NULL AND 
                r.{source_property} <> '' AND 
                r.embedding IS NULL
            RETURN 
                r.uuid AS uuid, 
                r.{source_property} AS txt
        """
        
        records = list(runner(query))
        
        # Generate embeddings for each record
        for record in records:
            vec = create_embedding(input_text=record["txt"])
            
            # Update relationship with embedding
            update_query = f"""
                MATCH ()-[r:{rel_type} {{uuid: $uuid}}]-()
                SET r.embedding = $vec
            """
            runner(update_query, {"uuid": record["uuid"], "vec": vec})
            
            # Debug output
            print(f"  text: {record['txt']}\n  vec: {vec[:3]}")

def create_query_cql(query: str) -> str:
    """
    Convert a natural language query to Cypher Query Language (CQL).
    
    Args:
        query: Natural language query to convert
        
    Returns:
        CQL query string (currently returns empty string - TODO implementation)
        
    TODO: Implement query to CQL conversion logic
    """
    cql_query = ""
    return cql_query
    
def neo4j_KGRAG_search(
    runner: Callable[[str, Optional[Mapping[str, Any]]], Iterable],
    element: Literal["node", "relationship"],
    query: str,
    index: str, 
    source_property : str,
    top_k: int
) -> Dict[str, Any]:
    """
    Perform KG RAG retrieval: vector search + context preparation for agent consumption
    
    - Combines vector search with context formatting specifically for RAG (Retrieval-Augmented Generation) use cases.
    - Searches for the most similar nodes or relationships to a given query using Neo4j's vector index capabilities.
    
    Args:
        runner: Query execution function (session.run or kg.query)
        element: Type of element to search - "node" or "relationship"
        node_label: Label of the node to process
        rel_type: Type of the relationship to process
        query: Query text to find similar items for
        index: Name of the vector index to use for search
        source_property: Property containing the text content to retrieve
        top_k: Number of most similar results to return
        
    Returns:
        Dictionary containing structured RAG context ready for agent consumption

    """
    
    # (1) Generate embedding for the search query
    query_embedding = create_embedding(query)
    
    # (2) Build default retrieval query based on element type
    if element == "node":
        vector_search_query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding) 
            YIELD node, score
            WITH node, labels(node) as label, score
            RETURN 
                score,
                node[$source_property] as text,
                node {{.*, label: labels(node)}} AS properties_dict
            ORDER BY score DESC
        """
        
    elif element == "relationship":
        vector_search_query = f"""
            CALL db.index.vector.queryRelationships($index_name, $top_k, $query_embedding) 
            YIELD relationship, score
            WITH relationship, type(relationship) as type, score
            RETURN 
                score, 
                relationship[$source_property] as text,
                relationship {{.*, type: type(relationship)}} AS properties_dict
            ORDER BY score DESC
        """
    else:
        raise ValueError(f"Invalid element type: {element}. Must be 'node' or 'relationship'")
    
    # (3) Execute search query
    search_parameters = {
        "element" : element,
        "source_property": source_property,
        "index_name": index, 
        "top_k": top_k,
        "query_embedding": query_embedding
    }
    
    cprint(f"\nRunning vector search query", "green")
    raw_results = runner(vector_search_query, search_parameters)

    # (4) Process results for RAG consumption

    processed_results = []
    combined_context = ""
    
    for i, result in enumerate(list(raw_results)):
        
        result_dict = dict(result)

        score = round(result_dict.get('score', 0.0), 5)
        text_content = result_dict.get('text')
        filtered_properties = {
            k: v for k, v in result_dict.get('properties_dict').items()
            if k not in {source_property, "embedding"}
        }
        
        processed_result = {
            "index": i,
            "score": score,
            "text": text_content,
            "properties": filtered_properties,
        }
        
        processed_results.append(processed_result)
        combined_context += f"\n\n {text_content}"
    
    # Return structured context for agent
    structured_context = {
        "query": query,
        "total_results": len(processed_results),
        "search_results": processed_results,
        "combined_context": combined_context,
        "search_metadata": {
            "element": element,
            "source_property": source_property,
            "index_used": index,
            "top_k": top_k,
        }
    }
    
    return structured_context
    

