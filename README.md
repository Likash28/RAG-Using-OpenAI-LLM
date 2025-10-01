# 🧠 Depression RAG Assistant

A specialized Retrieval-Augmented Generation (RAG) system designed to provide accurate, evidence-based information about depression and mental health. Built with FastAPI, Gemini LLM, and a beautiful Glass UI interface.

## ✨ Features

### 🎯 **Depression-Focused AI Assistant**
- **Specialized Knowledge**: Exclusively handles depression and mental health queries
- **Evidence-Based Responses**: Provides accurate, scientifically validated information
- **Crisis Detection**: Automatically detects and responds to suicidal ideation
- **Professional Boundaries**: Maintains appropriate clinical boundaries

### 🎨 **Modern Glass UI Interface**
- **Beautiful Design**: Glass morphism design with gradient backgrounds
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Separated Display**: Clean separation between main responses and sentiment analysis
- **Real-time Status**: Live connection status and file upload progress

### 🔍 **Advanced RAG Capabilities**
- **Multi-Modal Support**: Handles text, PDF, and image documents
- **Vector Search**: Semantic search across uploaded documents
- **Context-Aware**: Uses relevant document context for accurate responses
- **Source Attribution**: Shows sources for all information provided

### 📊 **Sentiment Analysis**
- **User Query Analysis**: Analyzes emotional tone of incoming questions
- **Response Sentiment**: Identifies emotional tone of AI responses
- **Justification**: Explains reasoning behind chosen response tone
- **Visual Separation**: Clean display of sentiment analysis

### 🛡️ **Safety & Security**
- **Crisis Intervention**: Immediate emergency response for crisis situations
- **Topic Guardrails**: Polite redirection for non-depression topics
- **Professional Disclaimers**: Appropriate medical disclaimers
- **Secure Configuration**: No API keys in repository

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Gemini API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Likash28/RAG-Using-OpenAI-LLM.git
   cd RAG-Using-OpenAI-LLM
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env and add your Gemini API key
   ```

5. **Run the application**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

6. **Open in browser**
   ```
   http://localhost:8000
   ```

## 📁 Project Structure

```
├── app.py                 # FastAPI application
├── pipeline.py            # RAG pipeline with Gemini LLM
├── config.py              # Configuration management
├── gemini_llm.py          # Gemini LLM wrapper
├── vectorstore.py         # Vector database management
├── embedder.py            # Text and image embeddings
├── logging_config.py      # Centralized logging
├── prompts/               # Prompt system
│   ├── system_prompt.txt  # Depression-focused system prompt
│   └── loader.py          # Topic detection and crisis handling
├── static/                # Frontend files
│   ├── index.html         # Main interface
│   ├── style.css          # Glass UI styling
│   └── script.js          # Frontend logic
├── utils/                 # Utility functions
├── requirements.txt       # Python dependencies
├── env.example           # Environment template
└── .gitignore            # Git ignore rules
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Provider
PROVIDER=gemini

# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# Embeddings
TEXT_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
CLIP_EMBEDDER=clip-ViT-B-32

# Database
CHROMA_DIR=./chroma_db
SQLITE_PATH=./facts.db

# RAG Settings
TOP_K=5
MAX_TOKENS=600

# Logging
ENVIRONMENT=development
DEV_LOG_LEVEL=DEBUG
PROD_LOG_LEVEL=INFO
LOG_DIR=logs
```

## 🎯 Usage

### 1. **Upload Documents**
- Drag and drop files or click to browse
- Supports PDF, images, and text files
- Documents are automatically processed and indexed

### 2. **Ask Questions**
- Type depression-related questions in the chat interface
- Get evidence-based responses with source attribution
- View sentiment analysis for each interaction

### 3. **Crisis Support**
- Automatic detection of crisis situations
- Immediate emergency resources and helpline numbers
- Professional crisis intervention protocols

## 🔍 API Endpoints

### Health Check
```http
GET /health
```

### Upload Documents
```http
POST /ingest
Content-Type: multipart/form-data
```

### Ask Questions
```http
POST /ask
Content-Type: application/json
{
  "query": "What are the symptoms of depression?",
  "k": 5
}
```

### Reset Conversation
```http
POST /reset
```

## 🛡️ Safety Features

### Crisis Detection
- Automatically identifies suicidal ideation
- Provides immediate emergency resources
- Bypasses normal processing for safety

### Topic Guardrails
- Only responds to depression-related queries
- Polite redirection for off-topic questions
- Maintains professional boundaries

### Professional Disclaimers
- Clear AI assistant disclaimers
- Medical advice disclaimers
- Crisis resource information

## 📊 Sentiment Analysis

The system provides detailed sentiment analysis for every interaction:

- **User Query Sentiment**: Analyzes emotional tone of questions
- **Response Sentiment**: Identifies emotional tone of responses
- **Justification**: Explains reasoning behind chosen tone

## 🔧 Development

### Running in Development Mode
```bash
uvicorn app:app --reload --port 8000
```

### Logging
- Comprehensive logging system
- Request/response tracking
- Error monitoring
- Log files in `logs/` directory

### Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of depression?", "k": 3}'
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `prompts/` directory
- Review the logging files for debugging

## ⚠️ Important Notes

- **Not a Replacement for Professional Help**: This system provides information only, not medical advice
- **Crisis Situations**: Always seek immediate professional help for crisis situations
- **API Keys**: Never commit API keys to the repository
- **Professional Use**: Designed for informational purposes, not clinical use

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Integration with more LLM providers
- [ ] Mobile app development
- [ ] Enhanced crisis detection algorithms

---

**Built with ❤️ for mental health awareness and support, Do drop a STAR if you liked it.**
