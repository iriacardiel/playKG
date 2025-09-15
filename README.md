# playKG: Neo4j Hello World (WSL + Docker + Python)
A playground to understand Knowledge Graphs.

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

3. Run Python script to test trough terminal

```bash
python main.py
``` 

4. Open Neo4j browser at http://localhost:7474 and login with `neo4j/test1234` (or the password you set in `docker-compose.yml`)


