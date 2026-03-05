"""
Test suite for enhanced concept extraction with domain-specific models
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extraction.extractor import ConceptExtractor


class TestExtractionEnhancement:
    """Test enhanced concept extraction"""
    
    def __init__(self):
        self.test_results = []
    
    def run_all_tests(self):
        """Run all test cases"""
        logger.info("=" * 80)
        logger.info("Starting Enhanced Extraction Tests")
        logger.info("=" * 80)
        
        # Test 1: Scientific domain
        self.test_scientific_extraction()
        
        # Test 2: Biomedical domain
        self.test_biomedical_extraction()
        
        # Test 3: Computer Science domain
        self.test_cs_extraction()
        
        # Test 4: Pattern-based extraction
        self.test_pattern_extraction()
        
        # Print summary
        self.print_summary()
    
    def test_scientific_extraction(self):
        """Test extraction on scientific text"""
        test_name = "Scientific Domain Extraction"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            extractor = ConceptExtractor(domain="scientific", use_domain_models=True)
            
            text = """
            The Heisenberg uncertainty principle is a fundamental theorem in quantum mechanics.
            This principle states that the position and momentum of a particle cannot be 
            simultaneously measured with arbitrary precision. The mathematical formulation 
            involves the Planck constant and demonstrates wave-particle duality. Recent 
            experiments using quantum entanglement have further validated this hypothesis.
            """
            
            result = extractor.extract("test_chunk_1", "test_doc_1", text)
            
            logger.info(f"Extracted {len(result.entities)} entities:")
            for entity in result.entities[:5]:
                logger.info(f"  - {entity.text} ({entity.label}) [confidence: {entity.confidence:.2f}]")
            
            logger.info(f"\nExtracted {len(result.keyphrases)} keyphrases:")
            for kp in result.keyphrases[:10]:
                logger.info(f"  - {kp.phrase} [score: {kp.score:.3f}]")
            
            if len(result.entities) > 0 and len(result.keyphrases) > 0:
                self.test_results.append((test_name, "PASSED", f"{len(result.entities)} entities, {len(result.keyphrases)} keyphrases"))
            else:
                self.test_results.append((test_name, "FAILED", "No concepts extracted"))
                
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))

    def test_biomedical_extraction(self):
        """Test extraction on biomedical text"""
        test_name = "Biomedical Domain Extraction"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            extractor = ConceptExtractor(domain="biomedical", use_domain_models=True)
            
            text = """
            The BRCA1 gene is associated with increased risk of breast cancer and ovarian cancer.
            Mutations in this tumor suppressor gene can lead to hereditary breast-ovarian cancer syndrome.
            Treatment options include chemotherapy, radiation therapy, and targeted antibody therapy.
            The protein encoded by BRCA1 plays a critical role in DNA repair mechanisms.
            """
            
            result = extractor.extract("test_chunk_2", "test_doc_2", text)
            
            logger.info(f"Extracted {len(result.entities)} entities:")
            for entity in result.entities[:5]:
                logger.info(f"  - {entity.text} ({entity.label}) [confidence: {entity.confidence:.2f}]")
            
            logger.info(f"\nExtracted {len(result.keyphrases)} keyphrases:")
            for kp in result.keyphrases[:10]:
                logger.info(f"  - {kp.phrase} [score: {kp.score:.3f}]")
            
            if len(result.keyphrases) > 0:
                self.test_results.append((test_name, "PASSED", f"{len(result.entities)} entities, {len(result.keyphrases)} keyphrases"))
            else:
                self.test_results.append((test_name, "FAILED", "No concepts extracted"))
                
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_cs_extraction(self):
        """Test extraction on computer science text"""
        test_name = "Computer Science Domain Extraction"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            extractor = ConceptExtractor(domain="computer_science", use_domain_models=True)
            
            text = """
            Deep learning using Convolutional Neural Networks (CNN) has revolutionized computer vision.
            The Transformer architecture, introduced in the paper "Attention is All You Need", 
            uses self-attention mechanisms for sequence-to-sequence tasks. BERT and GPT models 
            have achieved state-of-the-art results in natural language processing. Training these 
            models requires significant computational resources and large datasets.
            """
            
            result = extractor.extract("test_chunk_3", "test_doc_3", text)
            
            logger.info(f"Extracted {len(result.entities)} entities:")
            for entity in result.entities[:5]:
                logger.info(f"  - {entity.text} ({entity.label}) [confidence: {entity.confidence:.2f}]")
            
            logger.info(f"\nExtracted {len(result.keyphrases)} keyphrases:")
            for kp in result.keyphrases[:10]:
                logger.info(f"  - {kp.phrase} [score: {kp.score:.3f}]")
            
            if len(result.keyphrases) > 0:
                self.test_results.append((test_name, "PASSED", f"{len(result.entities)} entities, {len(result.keyphrases)} keyphrases"))
            else:
                self.test_results.append((test_name, "FAILED", "No concepts extracted"))
                
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_pattern_extraction(self):
        """Test pattern-based concept extraction"""
        test_name = "Pattern-Based Extraction"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            extractor = ConceptExtractor(domain="scientific", use_domain_models=True)
            
            text = """
            The Pythagorean theorem states that in a right triangle, the square of the hypotenuse
            equals the sum of squares of the other two sides. Newton's law of universal gravitation
            describes the gravitational force between masses. The Doppler effect explains the change
            in frequency of waves. These fundamental principles form the basis of classical physics.
            """
            
            result = extractor.extract("test_chunk_4", "test_doc_4", text)
            
            logger.info(f"Extracted {len(result.keyphrases)} keyphrases:")
            for kp in result.keyphrases[:10]:
                logger.info(f"  - {kp.phrase} [score: {kp.score:.3f}]")
            
            # Check if patterns were detected
            pattern_phrases = [kp.phrase for kp in result.keyphrases if "theorem" in kp.phrase.lower() or "law" in kp.phrase.lower() or "effect" in kp.phrase.lower()]
            
            logger.info(f"\nPattern-based concepts found: {len(pattern_phrases)}")
            for phrase in pattern_phrases:
                logger.info(f"  - {phrase}")
            
            if len(pattern_phrases) > 0:
                self.test_results.append((test_name, "PASSED", f"Found {len(pattern_phrases)} pattern-based concepts"))
            else:
                self.test_results.append((test_name, "WARNING", "No pattern-based concepts found"))
                
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAILED")
        warnings = sum(1 for _, status, _ in self.test_results if status == "WARNING")
        total = len(self.test_results)
        
        for test_name, status, message in self.test_results:
            status_symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "⚠"
            logger.info(f"{status_symbol} {test_name}: {status} - {message}")
        
        logger.info("\n" + "-" * 80)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Warnings: {warnings}")
        logger.info(f"Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%")
        logger.info("=" * 80)


if __name__ == "__main__":
    tester = TestExtractionEnhancement()
    tester.run_all_tests()
