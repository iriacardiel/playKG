# playKG: Neo4j Hello World

Hi! This is a playground to understand Knowledge Graphs. This project demonstrates how to set up a simple Neo4j database using Docker, and interact with it using Python. It includes examples of creating nodes and relationships, as well as querying the graph.

The project implements the **semantic retrieval** component of a KG RAG pipeline, which enables natural language queries against a knowledge graph containing information about people and companies.

![alt text](media/KGRAG_schema.svg)

## Architecture

### Data Model
- **Entities**: Person and Company nodes with unique constraints
- **Relationships**: KNOWS (person-to-person) and WORKS_AT (person-to-company)
- **Properties**: Basic attributes (name, age, education, industry) plus rich text descriptions
- **Embeddings**: Vector representations of entity descriptions using `nomic-embed-text` model

### Pipeline Components

1. **Graph Setup**
   - Create constraints for data integrity
   - Populate sample data (5 people, 3 companies, relationships)
   - Add descriptive text properties for entities

2. **Vector Index Creation**
   - Create vector indexes for Person and Company nodes
   - Use 768-dimensional embeddings with cosine similarity
   - Index properties: `embedding`

3. **Embedding Generation**
   - Generate embeddings for entity descriptions using Ollama
   - Store embeddings as node properties in Neo4j
   - Only process nodes missing embeddings (incremental updates)

4. **Semantic Search**
   - Convert user queries to embeddings
   - Perform vector similarity search against indexed properties
   - Return ranked results with similarity scores

## Implementation Details

### Technologies Used
- **Neo4j**: Graph database for storing entities and relationships
- **Ollama**: Local embedding model (`nomic-embed-text`)
- **Python Libraries**: 
  - `neo4j` driver for direct database interaction
  - `langchain-neo4j` for simplified graph operations
  - Standard utilities (dotenv, yaml, termcolor)

### Two Implementation Approaches
1. **Native Neo4j Driver**: Direct database connection with session management
2. **LangChain Wrapper**: Simplified interface using `Neo4jGraph` class

### Query Examples
The system can handle semantic queries like:
- "Who shaved their head?" â†’ Returns Guillermo (highest similarity score)
- "Curly hair" â†’ Returns Gabriela and other hair-related descriptions
- Technical/industry terms â†’ Returns relevant company descriptions

## Current Capabilities

**Implemented:**
- Graph data modeling with constraints
- Vector embedding generation and storage
- Vector index creation and management
- Semantic similarity search
- Dual implementation approaches (native driver + LangChain)

**Future Extensions:**
- Query-to-Cypher translation for structured retrieval
- Information fusion (combining semantic + structured results)
- LLM response generation
- Complete end-to-end RAG pipeline

## Usage

### 1. Start Neo4j:

```bash
docker compose up -d
```

### 2. Install Python deps

```bash
python -m venv .venv
source .venv/bin/activate  
pip install uv
uv sync
```

### 3. Testing the setup

Run Jupyter notebook `neo4j_playground_friends.ipynb`. 

All Cypher queries are stored in `queries_friends.yaml` file, and the notebook loads them automatically to execute the following steps:

0. Create a graph database different from the default `neo4j` (requires Neo4j Enterprise Edition).
1. Create constraints to avoid duplicate nodes.
2. Populate the graph with sample data.
3. Query the graph to find people and their relationships.
4. Optionally, clean up the graph.


**Optional but recommended: Neo4j Desktop**

Interactively run Cypher queries. Open Neo4j browser at `http://localhost:${HTTP_PORT}` and login with the credentials in the `.env` file.

![alt text](media/neo4j_desktop_screenshot.png)

The system demonstrates how vector embeddings can enhance traditional graph queries by enabling semantic search over unstructured text properties within a structured knowledge graph.

# Learning Resources

- [Neo4j Cypher Cheat Sheet](https://neo4j.com/docs/cypher-cheat-sheet/5/all/)




