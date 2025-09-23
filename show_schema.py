from langchain_neo4j import Neo4jGraph
kg = Neo4jGraph(
    url="bolt://localhost:7687",          # or "neo4j+s://<host>:7687" for Aura
    username="neo4j",
    password="test1234"
)

# Refresh and print the LLM-friendly schema summary
kg.refresh_schema()
print(kg.schema)             # readable string: labels, rel-types, properties


