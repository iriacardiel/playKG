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
       
def create_vector_indexes(index_name:str="", node_label:Optional[str]="", relation_type:Optional[str]=""):
    
    try:
        if node_label:
            params = {'index_name': index_name, 'node_label': node_label}
            kg.query(queries["create_vector_index_nodes"], params)
        elif relation_type:
            params = {'index_name': index_name, 'relation_type': relation_type}
            kg.query(queries["create_vector_index_relations"], params)
    except Exception as e:
        print(f"An error ocurred {e}.")
            
    
        
create_vector_indexes(index_name="person_node_idx", node_label="Person")
create_vector_indexes(index_name="company_node_idx", node_label="Company")
create_vector_indexes(index_name="know_relationship_idx", relation_type="KNOWS")
    
# Show created vector indexes
results = kg.query("SHOW VECTOR INDEXES")
idx = list(results)
cprint(f"\nFound {len(idx)} vector index entries.", "green")
for r in idx:
    cprint("-"*20,"green")
    pprint(r)
    
    
cprint("\nCreate data", "green")
kg.query(queries["create_seed"])

# Add rich text descriptions 

cprint("\nQuery: Adding descriptions, appearance and summaries", "green")
for q in queries["add_text"]:
    kg.query(q)


# Add location property 

cprint("\nQuery: Adding location property", "green")
for q in queries["add_locations"]:
    kg.query(q)


# Show locations and plot maps
records = kg.query(queries["show_locations"])
# Replace with your query result rows
# records = [
#     {"name":"Iria","lat":40.437596,"lon":-3.711223,"labels":["Person"]},
#     {"name":"Guillermo","lat":40.455022,"lon":-3.692355,"labels":["Person"]},
#     {"name":"Gabriela","lat":40.475721,"lon":-3.711451,"labels":["Person"]},
#     {"name":"Paula","lat":40.490170,"lon":-3.654654,"labels":["Person"]},
#     {"name":"Cristina","lat":40.367462,"lon":-3.597745,"labels":["Person"]},
#     {"name":"Indra","lat":40.396648,"lon":-3.624635,"labels":["Company"]},
#     {"name":"CIEMAT","lat":40.453938,"lon":-3.728925,"labels":["Company"]},
#     {"name":"CBM","lat":40.549613,"lon":-3.690136,"labels":["Company"]},
# ]
for r in records:
    print(r)

# Follium map
helper_folium.create_map_from_rows(records)

# Leaflet map
helper_leaflet.create_map_from_rows(records)

    
# Create property embeddings 

# (p:PERSON): create embeddings only for nodes missing them
helper_neo4j.vectorize_property(runner = kg.query,
                   element = "node", 
                   node_label = "Person",
                   source_property = "text"
                   )

# (c:COMPANY): create embeddings only for nodes missing them
helper_neo4j.vectorize_property(runner = kg.query,
                   element = "node", 
                   node_label = "Company", 
                   source_property = "text",
                   )

# [r:KNOWS]: create embeddings only for nodes missing them
helper_neo4j.vectorize_property(runner = kg.query,
                   element = "relationship",
                   rel_type = "KNOWS",
                   source_property = "text"
                   )


#################################################################################
def create_person(person_name, person_age, person_gender, person_education, person_extra_text, person_latitude, person_longitude):
    param_dict = {'person_name':person_name,
            'person_age':person_age,
            'person_gender':person_gender,
            'person_education':person_education,
            'person_extra_text':person_extra_text,
            'person_latitude':person_latitude,
            'person_longitude':person_longitude
            }
    
    try:
        kg.query(queries["create_person"],
                param_dict)
        print(f"Succesfuly created person: {param_dict}")
    except Exception as e:
        print(f"An error ocurred: {e}")
        
    # (p:PERSON): create embeddings only for nodes missing them
    helper_neo4j.vectorize_property(runner = kg.query,
                   element = "node", 
                   node_label = "Person",
                   source_property = "text"
                   )
    
def create_company(company_name, company_industry, company_extra_text, company_latitude, company_longitude):
    param_dict = {'company_name':company_name,
            'company_industry':company_industry,
            'company_extra_text':company_extra_text,
            'company_latitude':company_latitude,
            'company_longitude':company_longitude
            }
    
    try:
        kg.query(queries["create_company"],
                param_dict)
        print(f"Succesfuly created company: {param_dict}")
    except Exception as e:
        print(f"An error ocurred: {e}")
    
    # (c:COMPANY): create embeddings only for nodes missing them
    helper_neo4j.vectorize_property(runner = kg.query,
                   element = "node", 
                   node_label = "Company",
                   source_property = "text"
                   )

def create_person_company_relation(person_name, company_name, since):
    param_dict = {'person_name':person_name,
            'company_name':company_name,
            'since':since
            }
    
    try:
        kg.query(queries["create_person_company_relation"],
                param_dict)
        print(f"Succesfuly created relation: {param_dict}")
    except Exception as e:
        print(f"An error ocurred: {e}")
 
def create_person_person_relation(start_person, end_person, knows_from, text):
    param_dict = {'start_person':start_person,
                'end_person':end_person,
                'knows_from':knows_from,
                'relation_text':text
                }
    
    try:
        kg.query(queries["create_person_person_relation"],
                param_dict)
        print(f"Succesfuly created relation: {param_dict}")
    except Exception as e:
        print(f"An error ocurred: {e}")
        
    # [r:KNOWS]: create embeddings only for nodes missing them
    helper_neo4j.vectorize_property(runner = kg.query,
                    element = "relationship",
                    rel_type = "KNOWS",
                    source_property = "text"
                    )

        
    
    
create_person("María", "40", "female", "Music", "She is the best dancer in the world.",40.0, -3.0)
create_company("LUMON", "Mistery","This company does not care much about labor rights.", 41.0, -3.0)
create_person_company_relation("María", "LUMON","1980")
create_person_person_relation("María", "Iciar", "salsa", "Both liked to go to salsa classes on wednesdays.")
    
