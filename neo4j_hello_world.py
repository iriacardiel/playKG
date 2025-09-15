from neo4j import GraphDatabase
import os
from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()  # Load local environment variables

# Connection details
URI = "bolt://localhost:" + os.environ.get("URI_PORT")
AUTH = (os.environ.get("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD"))
print(f"Connecting to Neo4j at {URI} with user {AUTH[0]} and password {AUTH[1]}")

driver = GraphDatabase.driver(URI, auth=AUTH)

def create_data(tx):
    tx.run("""
        CREATE (a:Person {name: "Iria"})
        CREATE (b:Person {name: "Guillermo"})
        CREATE (a)-[:KNOWS {since: 2025}]->(b)
    """)

def query_data(tx):
    result1 = tx.run("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, r.since, b.name")
    result2 = tx.run("MATCH (n:Person) RETURN n LIMIT 25")

    for record in result1:
        print(f"{record['a.name']} knows {record['b.name']} since {record['r.since']}")
        
    for record in result2:
        print(record["n"])

with driver.session() as session:
    session.execute_write(create_data)
    session.execute_read(query_data)
    
    # Optional cleanup: wipe all data in the DB (use with care!)
    wipe = True  # set True to delete everything
    if wipe:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")
    else:
        print("Cleanup skipped.")


driver.close()
