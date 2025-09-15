from neo4j import GraphDatabase

# Connection details
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "test1234")

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
    #session.execute_write(create_data)
    session.execute_read(query_data)

driver.close()
