from neo4j import GraphDatabase

uri = "neo4j://127.0.0.1:7687"
user = "neo4j"
password = "Yadav2480@"
database = "miniproject"

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=database) as session:
        # Count all nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        total_nodes = result.single()["count"]
        print(f"Total nodes in database: {total_nodes}")
        
        # Count by type
        result = session.run("MATCH (n) RETURN labels(n) as type, count(n) as count")
        print("\nNodes by type:")
        for record in result:
            print(f"  {record['type']}: {record['count']}")
        
        # Count relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        total_rels = result.single()["count"]
        print(f"\nTotal relationships: {total_rels}")
    
    driver.close()
    print("\n✓ Successfully connected to Neo4j!")
    
except Exception as e:
    print(f"✗ Error: {e}")
