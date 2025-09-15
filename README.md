# playKG: Neo4j Hello World (WSL + Docker + Python)
A playground to understand Knowledge Graphs.

# Summary

This project demonstrates how to set up a simple Neo4j database using Docker, and interact with it using Python. It includes examples of creating nodes and relationships, as well as querying the graph.

### Setup
1. Start Neo4j:

```bash
   docker compose up -d
```
2. Install Python deps

```bash
python -m venv .venv
source .venv/bin/activate  
pip install uv
uv sync
```

# Testing the setup

**Option 1**: Run Python script to test trough terminal

```bash
python neo4j_hello_world.py
``` 

**Option 2**: Run the Jupyter notebook `neo4j_hello_world.ipynb` to see examples of creating and querying data in Neo4j.

**Option 3**: Open Neo4j browser at http://localhost:${HTTP_PORT} and login with the credentials in the `.env` file.


