"""Test Neo4j connection"""
from src.graph.queries import GraphQueryEngine
from src.utils.config import load_config

try:
    config = load_config()
    print(f"Config loaded: {config}")
    print(f"Neo4j URI: {config.get('neo4j', {}).get('uri')}")
    print(f"Neo4j DB: {config.get('neo4j', {}).get('database')}")
    
    query_engine = GraphQueryEngine(config)
    print("✅ Neo4j Connected!")
    
    stats = query_engine.get_graph_statistics()
    print(f"Papers: {stats.get('total_papers')}")
    print(f"Concepts: {stats.get('total_concepts')}")
    
except Exception as e:
    print(f"❌ Neo4j Connection Failed: {e}")
    import traceback
    traceback.print_exc()
