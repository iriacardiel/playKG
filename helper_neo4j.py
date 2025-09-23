"""
Neo4j Vector Operations Module
"""

from typing import Literal, Optional, List, Dict, Any, Union
from neo4j._sync.work.session import Session
from langchain_neo4j import Neo4jGraph
from termcolor import cprint
  
from helper_ollama import create_embedding

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

def vectorize_property(
    element: Literal["node", "relationship"],
    label: str,
    source_property: str,
    session: Optional[Session] = None, 
    kg: Optional[Neo4jGraph] = None
) -> None:
    """
    Generate and store vector embeddings for specified property of nodes or relationships.
    
    This function processes all nodes or relationships of a given label that have
    non-empty values for the specified property and generates embeddings for them.
    Only processes items that don't already have embeddings.
    
    Args:
        element: Type of graph element - either "node" or "relationship"
        label: Label of the nodes or relationship type to process
        source_property: Name of the property to vectorize
        session: Optional Neo4j driver session for direct database access
        kg: Optional Neo4jGraph instance for Langchain-based access
        
    Raises:
        ValueError: If neither session nor kg is provided
        
    Note:
        Exactly one of session or kg must be provided.
    """
    
    # Determine the query runner based on available connection
    if session:
        runner = session.run  # Use Neo4j native driver session
    elif kg:
        runner = kg.query  # Use Neo4jGraph Langchain object
    else:
        raise ValueError("Either 'session' or 'kg' must be provided")
    
    if element == "node":
        _vectorize_nodes(runner, label, source_property)
    elif element == "relationship":
        _vectorize_relationships(runner, label, source_property)


def _vectorize_nodes(runner, label: str, source_property: str) -> None:
    """
    Helper function to vectorize node properties.
    
    Args:
        runner: Query execution function (session.run or kg.query)
        label: Node label to process
        source_property: Property name to vectorize
    """
    cprint(f"\nGenerating embeddings for (n:{label}) on n.{source_property}", "green")
    
    # Query for nodes without embeddings
    query = f"""
        MATCH (n:{label})
        WHERE n.{source_property} IS NOT NULL 
        AND n.{source_property} <> ''
        AND n.embedding IS NULL
        RETURN n.uuid AS uuid, n.{source_property} AS txt
    """
    
    records = list(runner(query))
    
    # Generate embeddings for each record
    for record in records:
        vec = create_embedding(input_text=record["txt"])
        
        # Update node with embedding
        update_query = f"""
            MATCH (n:{label} {{uuid: $uuid}})
            SET n.embedding = $vec
        """
        runner(update_query, {"uuid": record["uuid"], "vec": vec})


def _vectorize_relationships(runner, label: str, source_property: str) -> None:
    """
    Helper function to vectorize relationship properties.
    
    Args:
        runner: Query execution function (session.run or kg.query)
        label: Relationship type to process
        source_property: Property name to vectorize
    """
    cprint(f"\nGenerating embeddings for [r:{label}] on r.{source_property}", "green")
    
    # Query for relationships without embeddings
    query = f"""
        MATCH ()-[r:{label}]-()
        WHERE r.{source_property} IS NOT NULL 
        AND r.{source_property} <> ''
        AND r.embedding IS NULL
        RETURN r.uuid AS uuid, r.{source_property} AS txt
    """
    
    records = list(runner(query))
    
    # Generate embeddings for each record
    for record in records:
        vec = create_embedding(input_text=record["txt"])
        
        # Update relationship with embedding
        update_query = f"""
            MATCH ()-[r:{label} {{uuid: $uuid}}]-()
            SET r.embedding = $vec
        """
        runner(update_query, {"uuid": record["uuid"], "vec": vec})
        
        # Debug output
        print(f"  text: {record['txt']}\n  vec: {vec[:3]}")


def neo4j_vector_search(
    element: Literal["node", "relationship"] = "node", 
    query: str = "", 
    index: str = "", 
    top_k: int = 10,
    session: Optional[Session] = None, 
    kg: Optional[Neo4jGraph] = None,
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search on Neo4j nodes or relationships.
    
    This function searches for the most similar nodes or relationships to a given
    query using Neo4j's vector index capabilities.
    
    Args:
        element: Type of element to search - "node" or "relationship"
        query: Query text to find similar items for
        index: Name of the vector index to use for search
        session: Optional Neo4j driver session for direct database access
        kg: Optional Neo4jGraph instance for Langchain-based access
        top_k: Number of most similar results to return
        
    Returns:
        List of dictionaries containing search results with scores and element data
        
    Raises:
        ValueError: If neither session nor kg is provided
        
    Note:
        Exactly one of session or kg must be provided.
    """
    # Determine the query runner based on available connection
    if session:
        runner = session.run  # Use Neo4j native driver session
    elif kg:
        runner = kg.query  # Use Neo4jGraph Langchain object
    else:
        raise ValueError("Either 'session' or 'kg' must be provided")
    
    # Generate embedding for the search query
    query_embedding = create_embedding(query)
    
    # Build appropriate vector search query
    if element == "node":
        vector_search_query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding) 
            YIELD node, score
            {create_query_cql(query)}
            RETURN score, node {{.* , embedding: ""}}
        """
    elif element == "relationship":
        vector_search_query = f"""
            CALL db.index.vector.queryRelationships($index_name, $top_k, $query_embedding) 
            YIELD relationship, score
            {create_query_cql(query)}
            RETURN score, relationship {{.* , embedding: ""}}
            ORDER BY score DESC
        """
    else:
        raise ValueError(f"Invalid element type: {element}. Must be 'node' or 'relationship'")
    
    # Execute search query
    search_parameters = {
        "index_name": index, 
        "top_k": top_k,
        "query_embedding": query_embedding
    }
    
    result_cursor = runner(vector_search_query, search_parameters)
    results = list(result_cursor)
    
    return results
