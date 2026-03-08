from src.graph.builder import KnowledgeGraphBuilder
from src.utils.config import Config

config = Config()
builder = KnowledgeGraphBuilder(
    config.get('neo4j.uri'),
    config.get('neo4j.user'),
    config.get('neo4j.password'),
    config.get('neo4j.database')
)

result = builder.driver.execute_query('MATCH (n) RETURN count(n) as count')
print(f'Total nodes in Neo4j: {result.records[0]["count"]}')

result = builder.driver.execute_query('MATCH (p:Paper) RETURN count(p) as count')
print(f'Total Paper nodes: {result.records[0]["count"]}')

builder.close()
