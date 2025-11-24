#!/bin/bash

# MRAG-SAR Evaluation Runner Script
# This script provides easy access to all evaluation frameworks

set -e  # Exit on any error

echo "🎯 MRAG-SAR Evaluation Framework"
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if virtual environment is activated (optional)
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected (recommended to use one)"
fi

# Function to run evaluation
run_evaluation() {
    local eval_type=$1
    local description=$2
    
    echo ""
    echo "🚀 Running $description..."
    echo "----------------------------------------"
    
    if python3 run_evaluation.py --$eval_type; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed"
        return 1
    fi
}

# Function to show help
show_help() {
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  quick       Run quick test (2 test cases)"
    echo "  core        Run core metrics evaluation"
    echo "  sentiment   Run sentiment evaluation"
    echo "  user        Run user studies"
    echo "  all         Run all evaluations (comprehensive)"
    echo "  test        Run quick test example"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick     # Quick test to verify setup"
    echo "  $0 core      # Core metrics evaluation"
    echo "  $0 all       # Full comprehensive evaluation"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "quick")
        run_evaluation "quick" "Quick Evaluation"
        ;;
    "core")
        run_evaluation "core" "Core Metrics Evaluation"
        ;;
    "sentiment")
        run_evaluation "sentiment" "Sentiment Evaluation"
        ;;
    "user")
        run_evaluation "user-studies" "User Studies"
        ;;
    "all")
        run_evaluation "all" "Comprehensive Evaluation"
        ;;
    "test")
        echo ""
        echo "🧪 Running Quick Test Example..."
        echo "----------------------------------------"
        if python3 quick_test_example.py; then
            echo "✅ Quick test example completed successfully"
        else
            echo "❌ Quick test example failed"
            exit 1
        fi
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "🎉 Evaluation completed!"
echo "📁 Check the generated JSON files for detailed results"
echo "📖 See EVALUATION_README.md for more information"
