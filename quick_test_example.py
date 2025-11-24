#!/usr/bin/env python3
"""
Quick Test Example for MRAG-SAR Evaluation Framework

This script demonstrates how to run a quick evaluation of the system
to verify everything is working correctly.

Usage:
    python quick_test_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from test import EnhancedTestSuite, print_enhanced_evaluation_summary
from logging_config import setup_logging, get_logger

async def run_quick_test():
    """Run a quick test to verify the evaluation framework works"""
    
    # Setup logging
    setup_logging()
    logger = get_logger("QuickTest")
    
    print("🚀 MRAG-SAR Quick Test Example")
    print("=" * 50)
    
    try:
        # Initialize enhanced test suite
        print("📝 Initializing enhanced test suite...")
        test_suite = EnhancedTestSuite()
        
        # Use only 2 test cases for quick testing
        test_suite.test_cases = test_suite.test_cases[:2]
        print(f"   Using {len(test_suite.test_cases)} test cases for quick evaluation")
        
        # Initialize pipeline
        print("🔧 Initializing RAG pipeline...")
        test_suite.initialize_pipeline()
        print("   ✅ Pipeline initialized successfully")
        
        # Run tests
        print("🧪 Running test cases...")
        results = await test_suite.run_all_tests()
        print(f"   ✅ Completed {len(results)} test cases")
        
        # Print enhanced summary
        print("\n📊 ENHANCED QUICK TEST RESULTS")
        print("-" * 30)
        print_enhanced_evaluation_summary(results)
        
        # Calculate aggregate metrics
        aggregate_metrics = test_suite.calculate_aggregate_metrics()
        
        print(f"\n🎯 KEY METRICS SUMMARY")
        print("-" * 30)
        print(f"Sentiment Alignment: {aggregate_metrics.sentiment_alignment_accuracy:.3f}")
        print(f"Overall CARE Score: {aggregate_metrics.overall_care_score:.3f}")
        print(f"Response Safety: {aggregate_metrics.response_safety_score:.3f}")
        print(f"Average Response Time: {aggregate_metrics.response_time_ms:.1f}ms")
        
        # Check if metrics look reasonable
        if aggregate_metrics.sentiment_alignment_accuracy > 0.5:
            print("\n✅ Sentiment alignment looks good!")
        else:
            print("\n⚠️  Sentiment alignment may need improvement")
        
        if aggregate_metrics.response_safety_score > 0.7:
            print("✅ Response safety looks good!")
        else:
            print("⚠️  Response safety may need improvement")
        
        print(f"\n🎉 Quick test completed successfully!")
        print(f"   To run full evaluation: python run_evaluation.py --all")
        print(f"   To run core metrics: python run_evaluation.py --core")
        print(f"   To run sentiment evaluation: python run_evaluation.py --sentiment")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        print(f"\n❌ Quick test failed: {e}")
        print(f"   Check your environment setup and API keys")
        print(f"   Make sure all dependencies are installed")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_quick_test())
    sys.exit(0 if success else 1)
