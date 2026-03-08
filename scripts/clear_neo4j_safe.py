"""
Safely clear Neo4j knowledge graph
"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("CLEARING NEO4J KNOWLEDGE GRAPH")
print("=" * 60)

try:
    # Connect to Neo4j
    uri = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    database = os.getenv('NEO4J_DATABASE', 'miniproject')
    
    print(f"\nConnecting to: {uri}")
    print(f"Database: {database}")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=database) as session:
        # Count nodes before deletion
        result = session.run("MATCH (n) RETURN count(n) as count")
        count_before = result.single()['count']
        print(f"\nNodes before deletion: {count_before}")
        
        # Delete all nodes and relationships
        session.run("MATCH (n) DETACH DELETE n")
        
        # Count nodes after deletion
        result = session.run("MATCH (n) RETURN count(n) as count")
        count_after = result.single()['count']
        print(f"Nodes after deletion: {count_after}")
    
    driver.close()
    
    print("\n" + "=" * 60)
    print("NEO4J CLEARED SUCCESSFULLY")
    print("=" * 60)
    print("\nAll knowledge graph data has been removed.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nMake sure:")
    print("1. Neo4j is running")
    print("2. Credentials in .env are correct")
    print("3. Database 'miniproject' exists")
