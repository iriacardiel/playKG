# Dependencies

import os
from dotenv import load_dotenv  
import yaml
from pathlib import Path
from pprint import pprint
from termcolor import cprint
from langchain_neo4j import Neo4jGraph
import json
from typing import Optional


from helpers import helper_folium, helper_leaflet, helper_neo4j

# Environment variables

load_dotenv()  # Load local environment variables

URI = "bolt://localhost:" + os.environ.get("URI_PORT")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PWD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")    # 👈 choose DB here


cprint(f"Connecting to Neo4j at {URI} with user {NEO4J_USER} and password {NEO4J_PWD}", "green")

# Load cypher queries

queries = yaml.safe_load(Path("data/friends/queries_friends.yaml").read_text())

# Neo4j Langchain wrapper instance

kg = Neo4jGraph(url=URI, username=NEO4J_USER, password=NEO4J_PWD, database=NEO4J_DB)


# Populate graph

cprint(f"\nConnected to Neo4j database: {NEO4J_DB}", "green")

cprint("\nCreating constraints (if not exist)", "green")
for q in queries["constraints"]:
    kg.query(q)
    
cprint("\nInit Cleanup.", "green")
for q in queries["delete_all"]:
    kg.query(q)

# Create vector indexes
helper_neo4j.create_vector_index(kg, index_name="person_node_idx", node_label="Person")
helper_neo4j.create_vector_index(kg, index_name="company_node_idx", node_label="Company")
helper_neo4j.create_vector_index(kg, index_name="know_relationship_idx", relation_type="KNOWS")
helper_neo4j.show_vector_indexes(kg)


with open("data/friends/friends.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# People
for p in data["people"]:
    
    helper_neo4j.create_node(kg=kg,
        label="Person",
        props={
            "name": p.get("name",""),
            "age": p.get("age",0),
            "gender": p.get("gender",""),
            "education": p.get("education",""),
            "latitude": p.get("latitude",0.0),
            "longitude":p.get("longitude",0.0),
        },
        text_template=f"{p.get("name","")} is a {p.get("age",0)} of {p.get("gender","")} years old and studied {p.get("education","")}. {p.get("extra_text", "")}",
        location_keys=("latitude", "longitude"),
        vectorize=True,
        source_property="text",
    )

# Companies
for c in data["companies"]:

    helper_neo4j.create_node(kg=kg,
        label="Company",
        props={
            "name": c.get("name",""),
            "industry": c.get("industry",0),
            "latitude": c.get("latitude",0.0),
            "longitude":c.get("longitude",0.0),
        },
        text_template=f"{c.get("name","")} operates in the {c.get("industry",0)} Industry. {c.get("extra_text", "")}",
        location_keys=("latitude", "longitude"),
        vectorize=True,
        source_property="text",
    )


# Person ↔ Person relationships
for r in data["person_person_relations"]:

    helper_neo4j.create_relationship(kg=kg,
        start_label="Person",
        end_label="Person",
        rel_type="KNOWS",
        start_value=r.get("start_person",""),
        end_value=r.get("end_person",""),
        rel_props={"knows_from": r.get("knows_from",""), "text": r.get("text","")},
        vectorize=True,
        source_property="text",
    )
    

# Person ↔ Company relationships
for r in data["person_company_relations"]:
    
    helper_neo4j.create_relationship(kg=kg,
        start_label="Person",
        end_label="Company",
        rel_type="WORKS_AT",
        start_key="name",
        end_key="name",
        start_value=r.get("person",""),
        end_value=r.get("company",""),
        rel_props={"since": r.get("since",0)},
    )

# Plot nodes into maps
helper_neo4j.create_visualizations(kg=kg,directory="data/friends/friends_map_")