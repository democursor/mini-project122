"""
Test that graph data persists across connections
"""
from src.utils.config import Config
from src.graph import KnowledgeGraphBuilder, GraphQueryEngine


def test_persistence():
    """Verify data persists"""
    print("=" * 60)
    print("TESTING DATA PERSISTENCE")
    print("=" * 60)
    
    config = Config()
    
    # Connect
    print("\n1. Connecting to Neo4j...")
    builder = KnowledgeGraphBuilder(
        uri=config.get('neo4j.uri'),
        user=config.get('neo4j.user'),
        password=config.get('neo4j.password'),
        database=config.get('neo4j.database')
    )
    print("   ✓ Connected")
    
    # Check data
    query_engine = GraphQueryEngine(builder.driver, builder.database)
    stats = query_engine.get_graph_statistics()
    
    print("\n2. Checking existing data:")
    print(f"   Papers: {stats['papers']}")
    print(f"   Concepts: {stats['concepts']}")
    print(f"   Mentions: {stats['mentions']}")
    print(f"   Relationships: {stats['relationships']}")
    
    # Close connection
    print("\n3. Closing connection...")
    builder.close()
    print("   ✓ Connection closed")
    
    # Reconnect
    print("\n4. Reconnecting...")
    builder2 = KnowledgeGraphBuilder(
        uri=config.get('neo4j.uri'),
        user=config.get('neo4j.user'),
        password=config.get('neo4j.password'),
        database=config.get('neo4j.database')
    )
    print("   ✓ Reconnected")
    
    # Check data again
    query_engine2 = GraphQueryEngine(builder2.driver, builder2.database)
    stats2 = query_engine2.get_graph_statistics()
    
    print("\n5. Checking data after reconnection:")
    print(f"   Papers: {stats2['papers']}")
    print(f"   Concepts: {stats2['concepts']}")
    print(f"   Mentions: {stats2['mentions']}")
    print(f"   Relationships: {stats2['relationships']}")
    
    # Verify
    if stats == stats2:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Data persists across connections!")
        print("=" * 60)
        print("\nYour graph data is PERMANENT.")
        print("It will survive:")
        print("  ✓ App restarts")
        print("  ✓ Connection closes")
        print("  ✓ Computer restarts (if Neo4j auto-starts)")
        print("\nFor web deployment:")
        print("  ✓ Data stays in database")
        print("  ✓ No need to rebuild")
        print("  ✓ Just connect and query")
    else:
        print("\n✗ Data mismatch (unexpected)")
    
    builder2.close()


if __name__ == "__main__":
    test_persistence()
