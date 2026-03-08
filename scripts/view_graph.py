"""
View knowledge graph data from command line
"""
from src.utils.config import Config
from src.graph import KnowledgeGraphBuilder, GraphQueryEngine


def view_graph():
    """Display graph data in terminal"""
    print("=" * 60)
    print("KNOWLEDGE GRAPH VIEWER")
    print("=" * 60)
    
    # Connect to Neo4j
    config = Config()
    builder = KnowledgeGraphBuilder(
        uri=config.get('neo4j.uri'),
        user=config.get('neo4j.user'),
        password=config.get('neo4j.password'),
        database=config.get('neo4j.database', 'neo4j')
    )
    
    query_engine = GraphQueryEngine(builder.driver, builder.database)
    
    # Get statistics
    print("\n📊 GRAPH STATISTICS:")
    stats = query_engine.get_graph_statistics()
    print(f"   Papers: {stats['papers']}")
    print(f"   Concepts: {stats['concepts']}")
    print(f"   Mentions: {stats['mentions']}")
    print(f"   Relationships: {stats['relationships']}")
    
    # Get all papers
    print("\n📄 ALL PAPERS:")
    papers = query_engine.get_all_papers(limit=10)
    for i, paper in enumerate(papers, 1):
        print(f"   {i}. {paper['title'][:60]}...")
        print(f"      Pages: {paper['page_count']}")
    
    # Get top concepts
    print("\n🏷️  TOP 20 CONCEPTS:")
    concepts = query_engine.get_all_concepts(limit=20)
    for i, concept in enumerate(concepts, 1):
        print(f"   {i:2d}. {concept['name']:30s} ({concept['type']:10s}) - {concept['frequency']} mentions")
    
    # Find related papers
    if papers:
        paper_id = papers[0]['id']
        print(f"\n🔗 PAPERS RELATED TO '{papers[0]['title'][:40]}...':")
        related = query_engine.find_related_papers(paper_id, limit=5)
        if related:
            for rel in related:
                print(f"   - {rel['title'][:60]}...")
                print(f"     Shared concepts: {rel['shared_concepts']}")
                print(f"     Sample: {', '.join(rel['sample_concepts'][:3])}")
        else:
            print("   No related papers found")
    
    # Find papers by concept
    if concepts:
        concept_name = concepts[0]['name']
        print(f"\n📑 PAPERS MENTIONING '{concept_name}':")
        papers_with_concept = query_engine.find_papers_by_concept(concept_name)
        for paper in papers_with_concept[:5]:
            print(f"   - {paper['title'][:60]}...")
            print(f"     Mentions: {paper['mentions']}, Confidence: {paper['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("To explore visually, open: http://localhost:7474")
    print("Login: neo4j / Yadav2480@")
    print("=" * 60)
    
    builder.close()


if __name__ == "__main__":
    view_graph()
