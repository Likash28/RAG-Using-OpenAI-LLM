"""
Comprehensive Evaluation Runner for MRAG-SAR

This script provides a unified interface to run all evaluation frameworks:
1. Core metrics evaluation (test.py)
2. Sentiment-aware prompting evaluation (sentiment_evaluation.py)
3. User studies and expert validation (user_studies.py)
4. Comprehensive reporting and analysis

Usage:
    python run_evaluation.py --all                    # Run all evaluations
    python run_evaluation.py --core                   # Run core metrics only
    python run_evaluation.py --sentiment              # Run sentiment evaluation only
    python run_evaluation.py --user-studies           # Run user studies only
    python run_evaluation.py --quick                  # Run quick evaluation subset
"""

import asyncio
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import evaluation modules
from test import TestSuite, print_evaluation_summary
from sentiment_evaluation import SentimentEvaluationRunner
from user_studies import run_example_user_study, generate_study_report
from logging_config import get_logger, setup_logging

logger = get_logger("EvaluationRunner")

class ComprehensiveEvaluationRunner:
    """Main runner for all evaluation frameworks"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_core_metrics_evaluation(self) -> Dict[str, Any]:
        """Run core metrics evaluation"""
        logger.info("Starting core metrics evaluation")
        
        test_suite = TestSuite()
        results = await test_suite.run_all_tests()
        
        # Calculate aggregate metrics
        aggregate_metrics = test_suite.calculate_aggregate_metrics()
        
        # Generate benchmark comparison
        benchmark_comparison = test_suite.generate_benchmark_comparison()
        
        # Generate ablation study
        ablation_study = test_suite.generate_ablation_study()
        
        # Generate error analysis
        error_analysis = test_suite.generate_error_analysis()
        
        core_results = {
            "evaluation_type": "core_metrics",
            "timestamp": datetime.now().isoformat(),
            "test_cases_count": len(test_suite.test_cases),
            "aggregate_metrics": aggregate_metrics.__dict__,
            "benchmark_comparison": [b.__dict__ for b in benchmark_comparison],
            "ablation_study": {k: v.__dict__ for k, v in ablation_study.items()},
            "error_analysis": error_analysis,
            "detailed_results": []
        }
        
        # Add detailed results (simplified)
        for test_case, metrics, detailed_result in results:
            result_entry = {
                "query": test_case.query,
                "expected_sentiment": test_case.expected_sentiment,
                "expected_response_type": test_case.expected_response_type,
                "safety_level": test_case.safety_level,
                "metrics": metrics.__dict__,
                "response_time_ms": detailed_result.get("response_time_ms", 0),
                "context_count": len(detailed_result.get("contexts", []))
            }
            core_results["detailed_results"].append(result_entry)
        
        return core_results
    
    async def run_sentiment_evaluation(self) -> Dict[str, Any]:
        """Run sentiment-aware prompting evaluation"""
        logger.info("Starting sentiment evaluation")
        
        runner = SentimentEvaluationRunner()
        results = await runner.run_comprehensive_sentiment_evaluation()
        
        return results
    
    async def run_user_studies(self) -> Dict[str, Any]:
        """Run user studies and expert validation"""
        logger.info("Starting user studies")
        
        # Run example user study (in real implementation, this would be actual user studies)
        user_study_results = await run_example_user_study()
        
        return user_study_results
    
    async def run_quick_evaluation(self) -> Dict[str, Any]:
        """Run a quick subset of evaluations for testing"""
        logger.info("Starting quick evaluation")
        
        # Run a smaller subset of tests
        test_suite = TestSuite()
        # Use only first 3 test cases for quick evaluation
        test_suite.test_cases = test_suite.test_cases[:3]
        
        results = await test_suite.run_all_tests()
        aggregate_metrics = test_suite.calculate_aggregate_metrics()
        
        quick_results = {
            "evaluation_type": "quick_evaluation",
            "timestamp": datetime.now().isoformat(),
            "test_cases_count": len(test_suite.test_cases),
            "aggregate_metrics": aggregate_metrics.__dict__,
            "note": "Quick evaluation with subset of test cases"
        }
        
        return quick_results
    
    async def run_all_evaluations(self) -> Dict[str, Any]:
        """Run all evaluation frameworks"""
        logger.info("Starting comprehensive evaluation")
        
        all_results = {
            "evaluation_metadata": {
                "start_time": datetime.now().isoformat(),
                "evaluation_type": "comprehensive",
                "framework_version": "1.0.0"
            },
            "evaluations": {}
        }
        
        # Run core metrics evaluation
        try:
            core_results = await self.run_core_metrics_evaluation()
            all_results["evaluations"]["core_metrics"] = core_results
            logger.info("Core metrics evaluation completed")
        except Exception as e:
            logger.error(f"Core metrics evaluation failed: {e}")
            all_results["evaluations"]["core_metrics"] = {"error": str(e)}
        
        # Run sentiment evaluation
        try:
            sentiment_results = await self.run_sentiment_evaluation()
            all_results["evaluations"]["sentiment_analysis"] = sentiment_results
            logger.info("Sentiment evaluation completed")
        except Exception as e:
            logger.error(f"Sentiment evaluation failed: {e}")
            all_results["evaluations"]["sentiment_analysis"] = {"error": str(e)}
        
        # Run user studies
        try:
            user_study_results = await self.run_user_studies()
            all_results["evaluations"]["user_studies"] = user_study_results
            logger.info("User studies completed")
        except Exception as e:
            logger.error(f"User studies failed: {e}")
            all_results["evaluations"]["user_studies"] = {"error": str(e)}
        
        # Generate comprehensive analysis
        all_results["comprehensive_analysis"] = self._generate_comprehensive_analysis(all_results)
        
        all_results["evaluation_metadata"]["end_time"] = datetime.now().isoformat()
        all_results["evaluation_metadata"]["duration_seconds"] = (
            datetime.now() - datetime.fromisoformat(all_results["evaluation_metadata"]["start_time"])
        ).total_seconds()
        
        return all_results
    
    def _generate_comprehensive_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis across all evaluations"""
        analysis = {
            "overall_performance": {},
            "key_insights": [],
            "recommendations": [],
            "strengths": [],
            "areas_for_improvement": [],
            "safety_assessment": {},
            "comparative_analysis": {}
        }
        
        # Extract key metrics from core evaluation
        if "core_metrics" in all_results["evaluations"] and "error" not in all_results["evaluations"]["core_metrics"]:
            core_metrics = all_results["evaluations"]["core_metrics"]["aggregate_metrics"]
            
            analysis["overall_performance"] = {
                "sentiment_alignment_accuracy": core_metrics.get("sentiment_alignment_accuracy", 0),
                "overall_care_score": core_metrics.get("overall_care_score", 0),
                "factual_consistency_score": core_metrics.get("factual_consistency_score", 0),
                "response_safety_score": core_metrics.get("response_safety_score", 0),
                "crisis_detection_accuracy": core_metrics.get("crisis_detection_accuracy", 0),
                "average_response_time_ms": core_metrics.get("response_time_ms", 0)
            }
        
        # Extract sentiment analysis insights
        if "sentiment_analysis" in all_results["evaluations"] and "error" not in all_results["evaluations"]["sentiment_analysis"]:
            sentiment_metrics = all_results["evaluations"]["sentiment_analysis"]["sentiment_metrics"]
            
            analysis["key_insights"].append(f"Sentiment alignment accuracy: {sentiment_metrics.get('sentiment_alignment_accuracy', 0):.3f}")
            analysis["key_insights"].append(f"Emotional appropriateness: {sentiment_metrics.get('emotional_appropriateness_score', 0):.3f}")
            analysis["key_insights"].append(f"Empathy score: {sentiment_metrics.get('empathy_score', 0):.3f}")
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(all_results)
        
        # Identify strengths and areas for improvement
        analysis["strengths"], analysis["areas_for_improvement"] = self._identify_strengths_and_weaknesses(all_results)
        
        # Safety assessment
        analysis["safety_assessment"] = self._assess_safety(all_results)
        
        return analysis
    
    def _generate_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Core metrics recommendations
        if "core_metrics" in all_results["evaluations"] and "error" not in all_results["evaluations"]["core_metrics"]:
            core_metrics = all_results["evaluations"]["core_metrics"]["aggregate_metrics"]
            
            if core_metrics.get("sentiment_alignment_accuracy", 0) < 0.7:
                recommendations.append("Improve sentiment alignment accuracy in responses")
            
            if core_metrics.get("crisis_detection_accuracy", 0) < 0.8:
                recommendations.append("Enhance crisis detection and response mechanisms")
            
            if core_metrics.get("response_safety_score", 0) < 0.8:
                recommendations.append("Strengthen safety measures and appropriateness checks")
        
        # Sentiment analysis recommendations
        if "sentiment_analysis" in all_results["evaluations"] and "error" not in all_results["evaluations"]["sentiment_analysis"]:
            sentiment_recs = all_results["evaluations"]["sentiment_analysis"].get("recommendations", [])
            recommendations.extend(sentiment_recs)
        
        return recommendations
    
    def _identify_strengths_and_weaknesses(self, all_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify system strengths and areas for improvement"""
        strengths = []
        weaknesses = []
        
        # Analyze core metrics
        if "core_metrics" in all_results["evaluations"] and "error" not in all_results["evaluations"]["core_metrics"]:
            core_metrics = all_results["evaluations"]["core_metrics"]["aggregate_metrics"]
            
            # Strengths
            if core_metrics.get("response_safety_score", 0) > 0.8:
                strengths.append("Strong safety measures and appropriateness")
            
            if core_metrics.get("factual_consistency_score", 0) > 0.7:
                strengths.append("Good factual consistency and grounding")
            
            if core_metrics.get("response_time_ms", 0) < 2000:
                strengths.append("Fast response times")
            
            # Weaknesses
            if core_metrics.get("sentiment_alignment_accuracy", 0) < 0.6:
                weaknesses.append("Sentiment alignment needs improvement")
            
            if core_metrics.get("empathy_score", 0) < 0.6:
                weaknesses.append("Empathy in responses could be enhanced")
        
        return strengths, weaknesses
    
    def _assess_safety(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall safety of the system"""
        safety_assessment = {
            "overall_safety_score": 0.0,
            "crisis_handling": "unknown",
            "safety_concerns": [],
            "recommendations": []
        }
        
        # Analyze safety from core metrics
        if "core_metrics" in all_results["evaluations"] and "error" not in all_results["evaluations"]["core_metrics"]:
            core_metrics = all_results["evaluations"]["core_metrics"]["aggregate_metrics"]
            
            safety_score = core_metrics.get("response_safety_score", 0)
            crisis_accuracy = core_metrics.get("crisis_detection_accuracy", 0)
            
            safety_assessment["overall_safety_score"] = (safety_score + crisis_accuracy) / 2
            
            if crisis_accuracy > 0.8:
                safety_assessment["crisis_handling"] = "excellent"
            elif crisis_accuracy > 0.6:
                safety_assessment["crisis_handling"] = "good"
            elif crisis_accuracy > 0.4:
                safety_assessment["crisis_handling"] = "fair"
            else:
                safety_assessment["crisis_handling"] = "poor"
                safety_assessment["safety_concerns"].append("Crisis detection accuracy is low")
        
        return safety_assessment
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of results"""
        print("\n" + "="*80)
        print("🎯 MRAG-SAR COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        # Evaluation metadata
        metadata = results.get("evaluation_metadata", {})
        print(f"\n📅 Evaluation Date: {metadata.get('start_time', 'Unknown')}")
        print(f"⏱️  Duration: {metadata.get('duration_seconds', 0):.1f} seconds")
        print(f"🔧 Framework Version: {metadata.get('framework_version', 'Unknown')}")
        
        # Overall performance
        if "comprehensive_analysis" in results:
            analysis = results["comprehensive_analysis"]
            overall_perf = analysis.get("overall_performance", {})
            
            print(f"\n📊 OVERALL PERFORMANCE")
            print("-" * 50)
            for metric, value in overall_perf.items():
                if isinstance(value, float):
                    print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Key insights
        if "comprehensive_analysis" in results:
            insights = analysis.get("key_insights", [])
            if insights:
                print(f"\n💡 KEY INSIGHTS")
                print("-" * 50)
                for insight in insights:
                    print(f"• {insight}")
        
        # Strengths and weaknesses
        if "comprehensive_analysis" in results:
            strengths = analysis.get("strengths", [])
            weaknesses = analysis.get("areas_for_improvement", [])
            
            if strengths:
                print(f"\n✅ STRENGTHS")
                print("-" * 50)
                for strength in strengths:
                    print(f"• {strength}")
            
            if weaknesses:
                print(f"\n⚠️  AREAS FOR IMPROVEMENT")
                print("-" * 50)
                for weakness in weaknesses:
                    print(f"• {weakness}")
        
        # Recommendations
        if "comprehensive_analysis" in results:
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                print(f"\n🎯 RECOMMENDATIONS")
                print("-" * 50)
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
        
        # Safety assessment
        if "comprehensive_analysis" in results:
            safety = analysis.get("safety_assessment", {})
            if safety:
                print(f"\n🛡️  SAFETY ASSESSMENT")
                print("-" * 50)
                print(f"Overall Safety Score: {safety.get('overall_safety_score', 0):.3f}")
                print(f"Crisis Handling: {safety.get('crisis_handling', 'Unknown').title()}")
                
                concerns = safety.get('safety_concerns', [])
                if concerns:
                    print("Safety Concerns:")
                    for concern in concerns:
                        print(f"  • {concern}")

async def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="MRAG-SAR Comprehensive Evaluation Runner")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--core", action="store_true", help="Run core metrics evaluation only")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment evaluation only")
    parser.add_argument("--user-studies", action="store_true", help="Run user studies only")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation subset")
    parser.add_argument("--output", type=str, help="Output filename for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    runner = ComprehensiveEvaluationRunner()
    
    try:
        if args.all:
            print("🚀 Running comprehensive evaluation (all frameworks)")
            results = await runner.run_all_evaluations()
        elif args.core:
            print("📊 Running core metrics evaluation")
            results = await runner.run_core_metrics_evaluation()
        elif args.sentiment:
            print("🧠 Running sentiment evaluation")
            results = await runner.run_sentiment_evaluation()
        elif args.user_studies:
            print("👥 Running user studies")
            results = await runner.run_user_studies()
        elif args.quick:
            print("⚡ Running quick evaluation")
            results = await runner.run_quick_evaluation()
        else:
            print("❌ No evaluation type specified. Use --help for options.")
            return
        
        # Print summary
        runner.print_summary(results)
        
        # Save results
        filename = runner.save_results(results, args.output)
        print(f"\n✅ Evaluation completed! Results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
