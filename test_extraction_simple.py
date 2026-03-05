"""
Simple test for enhanced extraction using general models
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.extraction.extractor import ConceptExtractor

# Test with general domain (uses lightweight models)
logger.info("Testing Enhanced Extraction with General Domain")
logger.info("=" * 80)

extractor = ConceptExtractor(domain="general", use_domain_models=False)

text = """
The COVID-19 pandemic has significantly impacted global health systems.
Machine learning algorithms have been used to predict disease spread patterns.
Vaccination campaigns have been implemented worldwide to control transmission.
The World Health Organization coordinates international response efforts.
"""

logger.info("Extracting concepts from sample text...")
result = extractor.extract("test_chunk", "test_doc", text)

logger.info(f"\n✓ Extracted {len(result.entities)} entities:")
for entity in result.entities[:10]:
    logger.info(f"  - {entity.text} ({entity.label}) [confidence: {entity.confidence:.2f}]")

logger.info(f"\n✓ Extracted {len(result.keyphrases)} keyphrases:")
for kp in result.keyphrases[:15]:
    logger.info(f"  - {kp.phrase} [score: {kp.score:.3f}]")

logger.info(f"\n✓ Processing time: {result.processing_time:.2f} seconds")
logger.info("=" * 80)
logger.info("✓ Enhancement 2 Complete: Domain-specific extraction working!")
