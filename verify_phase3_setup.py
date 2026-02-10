"""
Verify Phase 3 setup is complete and correct
"""
import sys
from pathlib import Path


def check_files():
    """Check all required files exist"""
    print("=" * 60)
    print("Checking Phase 3 Files")
    print("=" * 60)
    
    required_files = [
        'src/graph/__init__.py',
        'src/graph/builder.py',
        'src/graph/models.py',
        'src/graph/queries.py',
        'test_phase3.py',
        'build_graph.py',
        'config/default.yaml',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if imports work"""
    print("\n" + "=" * 60)
    print("Checking Python Imports")
    print("=" * 60)
    
    imports_ok = True
    
    # Check graph module
    try:
        from src.graph import KnowledgeGraphBuilder, PaperNode, ConceptNode, GraphQueryEngine
        print("✓ src.graph imports successfully")
    except ImportError as e:
        print(f"✗ src.graph import failed: {e}")
        imports_ok = False
    
    # Check workflow
    try:
        from src.orchestration.workflow import DocumentProcessor
        print("✓ src.orchestration.workflow imports successfully")
    except ImportError as e:
        print(f"✗ src.orchestration.workflow import failed: {e}")
        imports_ok = False
    
    # Check neo4j
    try:
        import neo4j
        print(f"✓ neo4j library installed (version {neo4j.__version__})")
    except ImportError:
        print("✗ neo4j library not installed")
        print("  Run: pip install neo4j")
        imports_ok = False
    
    return imports_ok


def check_config():
    """Check configuration"""
    print("\n" + "=" * 60)
    print("Checking Configuration")
    print("=" * 60)
    
    try:
        from src.utils.config import Config
        config = Config()
        
        # Check Neo4j config
        if config.get('neo4j'):
            print("✓ Neo4j configuration found")
            print(f"  URI: {config.get('neo4j.uri')}")
            print(f"  User: {config.get('neo4j.user')}")
            print(f"  Database: {config.get('neo4j.database')}")
        else:
            print("✗ Neo4j configuration missing in config/default.yaml")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def check_neo4j_connection():
    """Check if Neo4j is accessible"""
    print("\n" + "=" * 60)
    print("Checking Neo4j Connection")
    print("=" * 60)
    
    try:
        from src.utils.config import Config
        from src.graph import KnowledgeGraphBuilder
        
        config = Config()
        
        builder = KnowledgeGraphBuilder(
            uri=config.get('neo4j.uri'),
            user=config.get('neo4j.user'),
            password=config.get('neo4j.password'),
            database=config.get('neo4j.database', 'neo4j')
        )
        
        print("✓ Connected to Neo4j successfully")
        builder.close()
        return True
        
    except Exception as e:
        print(f"✗ Cannot connect to Neo4j: {e}")
        print("\n" + "=" * 60)
        print("NEO4J NOT RUNNING - Installation Options")
        print("=" * 60)
        print("\n📌 RECOMMENDED FOR WINDOWS: Neo4j Desktop")
        print("   1. Download: https://neo4j.com/download/")
        print("   2. Install and launch Neo4j Desktop")
        print("   3. Create database with password: 'password'")
        print("   4. Click 'Start' button")
        print("   5. Run: python verify_phase3_setup.py")
        print("\n   See: INSTALL_NEO4J_WINDOWS.md for detailed guide")
        print("\n📌 ALTERNATIVE: Install Docker Desktop")
        print("   1. Download: https://www.docker.com/products/docker-desktop/")
        print("   2. Install and restart computer")
        print("   3. Run: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
        print("\n📌 WORK WITHOUT NEO4J:")
        print("   Phase 3 has graceful degradation - you can continue without Neo4j")
        print("   Graph construction will be skipped but other phases work fine")
        print("   See: PHASE3_WITHOUT_NEO4J.md")
        return False


def check_parsed_data():
    """Check if parsed data exists"""
    print("\n" + "=" * 60)
    print("Checking Parsed Data")
    print("=" * 60)
    
    parsed_dir = Path('./data/parsed')
    
    if not parsed_dir.exists():
        print("✗ data/parsed/ directory not found")
        return False
    
    json_files = list(parsed_dir.glob('*.json'))
    
    if not json_files:
        print("✗ No parsed JSON files found")
        print("  Run Phase 1 & 2 first to generate parsed data")
        return False
    
    print(f"✓ Found {len(json_files)} parsed documents")
    
    # Check if any have concepts
    import json
    docs_with_concepts = 0
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get('concepts') and len(data.get('concepts', [])) > 0:
                docs_with_concepts += 1
    
    print(f"✓ {docs_with_concepts} documents have extracted concepts")
    
    if docs_with_concepts == 0:
        print("⚠ Warning: No documents have concepts extracted")
        print("  Phase 3 will work but graph will be empty")
    
    return True


def main():
    """Run all verification checks"""
    print("\n" + "=" * 60)
    print("PHASE 3 SETUP VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Files", check_files),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Neo4j Connection", check_neo4j_connection),
        ("Parsed Data", check_parsed_data)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("✓ PHASE 3 SETUP COMPLETE!")
        print("=" * 60)
        print("\nYou can now:")
        print("  1. Build graph: python build_graph.py")
        print("  2. Run tests: python test_phase3.py")
        print("  3. Process PDFs: python main.py")
        print("  4. Explore graph: http://localhost:7474")
        return 0
    else:
        print("\n" + "=" * 60)
        print("⚠ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease fix the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
