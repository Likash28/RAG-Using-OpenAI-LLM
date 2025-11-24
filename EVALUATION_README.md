# MRAG-SAR Comprehensive Evaluation Framework

This directory contains a comprehensive evaluation framework for the Multimodal RAG with Sentiment-Aware Responses (MRAG-SAR) system. The framework addresses all the metrics-related questions from your evaluation requirements.

## 🎯 Evaluation Coverage

This framework comprehensively covers all the metrics questions you listed:

### ✅ **Quantitative Evaluation of Sentiment-Aware Prompting**
- Sentiment alignment accuracy measurement
- Emotional appropriateness scoring
- Sentiment consistency evaluation
- Statistical significance testing

### ✅ **Multidimensional Evaluation Metrics**
- **Sentiment Alignment Accuracy**: Measures how well responses match query sentiment
- **CARE Scores**: Compassion, Accuracy, Relevance, Empathy evaluation
- **Factual Consistency**: Source citation accuracy and hallucination detection
- **Human Evaluation**: User studies and expert validation frameworks

### ✅ **Benchmarking Against Baselines**
- GPT-only system comparison
- Rule-based bot comparison
- Standard RAG system comparison
- Comparative performance tables

### ✅ **Ablation Studies**
- Modular performance breakdown (Whisper, BLIP, sentiment layer)
- Component-wise impact analysis
- Statistical significance testing

### ✅ **User Studies & Expert Validation**
- Mental health professional rating framework
- User experience evaluation
- A/B testing capabilities
- Qualitative feedback analysis

### ✅ **Error Analysis**
- Sentiment misalignment detection
- Retrieval failure analysis
- Safety violation identification
- Response quality issue tracking

## 📁 File Structure

```
├── test.py                    # Core metrics evaluation framework
├── sentiment_evaluation.py    # Sentiment-aware prompting evaluation
├── user_studies.py           # User studies and expert validation
├── run_evaluation.py         # Unified evaluation runner
├── test_requirements.txt     # Additional dependencies for testing
└── EVALUATION_README.md      # This documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install additional testing dependencies
pip install -r test_requirements.txt
```

### 2. Set Up Environment

Make sure your `.env` file is configured with:
```bash
GEMINI_API_KEY=your_api_key_here
# Other required environment variables
```

### 3. Run Evaluations

#### Run All Evaluations (Comprehensive)
```bash
python run_evaluation.py --all
```

#### Run Specific Evaluations
```bash
# Core metrics only
python run_evaluation.py --core

# Sentiment evaluation only
python run_evaluation.py --sentiment

# User studies only
python run_evaluation.py --user-studies

# Quick evaluation (subset for testing)
python run_evaluation.py --quick
```

#### Run Individual Modules
```bash
# Core metrics evaluation
python test.py

# Sentiment evaluation
python sentiment_evaluation.py

# User studies (example)
python user_studies.py
```

## 📊 Evaluation Modules

### 1. Core Metrics Evaluation (`test.py`)

**Purpose**: Comprehensive evaluation of all system metrics

**Key Features**:
- Sentiment alignment accuracy
- CARE scores (Compassion, Accuracy, Relevance, Empathy)
- Factual consistency and source citation accuracy
- Retrieval quality metrics (precision, recall, F1)
- Response quality (coherence, completeness, safety)
- Crisis detection accuracy
- Performance metrics (response time, token efficiency)

**Output**: Detailed metrics with statistical analysis

### 2. Sentiment Evaluation (`sentiment_evaluation.py`)

**Purpose**: Quantitative evaluation of sentiment-aware prompting

**Key Features**:
- Advanced sentiment analysis with intensity measurement
- Sentiment alignment accuracy by type (positive, negative, crisis, etc.)
- Emotional appropriateness scoring
- Sentiment consistency analysis
- Comparative analysis with/without sentiment awareness
- Statistical significance testing

**Output**: Specialized sentiment metrics and improvement analysis

### 3. User Studies (`user_studies.py`)

**Purpose**: User experience and expert validation

**Key Features**:
- User study framework with different user types
- Expert validation by mental health professionals
- A/B testing capabilities
- Qualitative feedback analysis
- Rating scales and statistical analysis
- Comprehensive reporting

**Output**: User feedback analysis and expert validation results

### 4. Unified Runner (`run_evaluation.py`)

**Purpose**: Orchestrate all evaluation frameworks

**Key Features**:
- Command-line interface
- Comprehensive reporting
- Cross-evaluation analysis
- Recommendations generation
- Safety assessment
- Results aggregation

## 📈 Metrics Explained

### Sentiment Analysis Metrics

- **Sentiment Alignment Accuracy**: How well system responses match the emotional tone of user queries
- **Emotional Appropriateness**: Whether responses are emotionally appropriate for the context
- **Sentiment Consistency**: Consistency of sentiment handling across similar queries

### CARE Scores

- **Compassion**: Presence of compassionate language and understanding
- **Accuracy**: Clinical and factual accuracy of information
- **Relevance**: Relevance to depression/mental health topics
- **Empathy**: Empathetic tone and validation of user feelings

### Safety Metrics

- **Crisis Detection Accuracy**: Ability to identify and appropriately respond to crisis situations
- **Response Safety Score**: Overall safety and appropriateness of responses
- **Professional Boundaries**: Maintenance of appropriate professional boundaries

### Performance Metrics

- **Response Time**: Average time to generate responses
- **Token Efficiency**: Tokens per second for response generation
- **Retrieval Quality**: Precision, recall, and F1 scores for document retrieval

## 🔬 Benchmarking

The framework includes comparison with:

1. **GPT-4 Only**: Baseline LLM without RAG
2. **Rule-Based Bot**: Traditional rule-based mental health bot
3. **Standard RAG**: Basic RAG without sentiment awareness
4. **MRAG-SAR**: Our system with full capabilities

### Comparative Metrics

- Modality support (text, image, audio)
- Emotional alignment capabilities
- Factual grounding accuracy
- Response time and cost
- Safety and appropriateness

## 🧪 Ablation Studies

The framework supports ablation studies for:

- **Sentiment Analysis Layer**: Impact of sentiment-aware prompting
- **Multimodal Support**: Effect of image and audio processing
- **Crisis Detection**: Importance of crisis detection mechanisms
- **Source Citations**: Impact of source citation on factual accuracy

## 👥 User Studies Framework

### User Types
- General users
- Mental health users
- Caregivers
- Students
- Professionals

### Expert Types
- Psychiatrists
- Psychologists
- Clinical social workers
- Counselors
- Nurse practitioners
- Researchers

### Rating Criteria
- Helpfulness
- Clarity
- Empathy
- Safety
- Trustworthiness
- Overall satisfaction

## 📋 Test Cases

The framework includes comprehensive test cases covering:

- **Depression-related queries**: Symptoms, treatment, coping strategies
- **Crisis situations**: Suicidal ideation, self-harm concerns
- **Off-topic queries**: Weather, cooking, general questions
- **Positive sentiment**: Recovery, progress, hope
- **Treatment questions**: Medications, therapy, professional help
- **Complex scenarios**: Multi-faceted mental health situations

## 📊 Output and Reporting

### JSON Reports
All evaluations generate detailed JSON reports with:
- Raw metrics and scores
- Statistical analysis
- Error analysis
- Recommendations
- Comparative data

### Summary Reports
Human-readable summaries with:
- Key performance indicators
- Strengths and weaknesses
- Recommendations
- Safety assessments

### Visualization (Future Enhancement)
- Performance dashboards
- Comparative charts
- Trend analysis
- User feedback visualization

## 🛡️ Safety and Ethics

The evaluation framework includes:

- **Safety Assessment**: Comprehensive safety evaluation
- **Crisis Handling**: Special evaluation for crisis situations
- **Professional Boundaries**: Assessment of appropriate boundaries
- **Ethical Considerations**: Framework for ethical evaluation

## 🔧 Customization

### Adding New Test Cases
```python
# In test.py, add to TestSuite._create_test_cases()
TestCase(
    query="Your test query here",
    expected_sentiment="expected_sentiment",
    expected_response_type="depression",
    expected_keywords=["keyword1", "keyword2"],
    safety_level="safe"
)
```

### Adding New Metrics
```python
# In test.py, add to EvaluationMetrics dataclass
new_metric: float = 0.0

# In MetricsEvaluator, implement evaluation logic
def evaluate_new_metric(self, response: str) -> float:
    # Your evaluation logic here
    return score
```

### Custom User Studies
```python
# In user_studies.py, create custom user profiles
custom_user = UserProfile(
    user_id="custom_001",
    user_type=UserType.CUSTOM,
    # ... other fields
)
```

## 📚 Dependencies

### Core Dependencies
- FastAPI, Pydantic, Uvicorn
- Google Generative AI
- ChromaDB, Sentence Transformers
- Unstructured, Pillow
- OpenAI Whisper (optional)

### Testing Dependencies
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- NLTK, SpaCy, TextStat
- Pytest, Pytest-asyncio

## 🚨 Important Notes

1. **API Keys**: Ensure all required API keys are configured
2. **Data Privacy**: User study data should be handled according to privacy regulations
3. **Expert Validation**: Real expert validation requires proper consent and IRB approval
4. **Crisis Handling**: Always prioritize safety in crisis situations
5. **Continuous Evaluation**: Run evaluations regularly to monitor system performance

## 🤝 Contributing

To add new evaluation capabilities:

1. Create new evaluation module following existing patterns
2. Add comprehensive test cases
3. Implement statistical analysis
4. Update documentation
5. Add to unified runner

## 📞 Support

For questions about the evaluation framework:
- Check the code comments and docstrings
- Review the example usage in each module
- Run the quick evaluation first to test setup
- Check logs for detailed error information

---

**Note**: This evaluation framework is designed to be comprehensive and rigorous. It addresses all the metrics questions from your research requirements and provides a solid foundation for evaluating the MRAG-SAR system's performance across multiple dimensions.
