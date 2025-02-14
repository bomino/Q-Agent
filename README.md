# Knowledge Base AI Assistant

## Overview
Knowledge Base AI Assistant is a Streamlit-based web application that enables users to analyze documents using AI models from OpenAI (GPT-3.5, GPT-4) and Anthropic (Claude). The application supports multiple file formats, handles large documents through intelligent chunking, and maintains conversation history.

## Features
- Support for multiple file formats (CSV, JSON, TXT)
- Integration with OpenAI (GPT-3.5, GPT-4) and Anthropic Claude
- Automatic content chunking for large documents
- Conversation history tracking and export
- Response caching for improved performance
- Visual progress indicators
- Export functionality for analysis results

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/knowledge-base-ai-assistant.git
cd knowledge-base-ai-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dependencies
Core dependencies:
```txt
streamlit>=1.32.0
openai>=1.12.0
anthropic>=0.18.0
pandas>=2.2.0
python-dotenv>=1.0.0
typing-extensions>=4.9.0
python-dateutil>=2.8.2
requests>=2.31.0
```

Development dependencies (optional):
```txt
pytest>=7.4.3
black>=24.1.1
flake8>=7.0.0
mypy>=1.8.0
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Basic workflow:
   - Enter your API key in the sidebar configuration
   - Upload knowledge base files (CSV, JSON, or TXT)
   - Enter your query in the text area
   - Click "Analyze" to process
   - View results and conversation history

## Project Structure
```
knowledge-base-ai-assistant/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables (create this)
├── logo.png          # Application logo
└── README.md             # This file
```

## Components

### FileProcessor
Handles file validation and processing:
- Maximum file size: 5MB
- Supported formats: CSV, JSON, TXT
- Validates file type and size
- Converts file content to structured strings

### ContentProcessor
Manages content chunking and token estimation:
- Handles model token limits
- Splits content into processable chunks
- Estimates token counts
- Manages different model constraints

### AIModelInterface
Manages AI model interactions:
- Handles OpenAI and Anthropic API calls
- Implements retry logic with exponential backoff
- Manages response caching
- Processes content chunks

### SessionState
Manages application state:
- Initializes session variables
- Tracks conversation history
- Manages file processing state

## Error Handling

The application includes comprehensive error handling for:
- Token limit exceeded errors
- API authentication issues
- File processing errors
- Network connectivity problems

Key features:
- Automatic retry with exponential backoff
- User-friendly error messages
- Detailed logging
- Graceful fallback mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch
```bash
git checkout -b feature/your-feature-name
```
3. Make your changes
4. Run tests
```bash
pytest
```
5. Format code
```bash
black .
flake8
mypy .
```
6. Submit a pull request

## Support
For support:
- Open an issue on GitHub
- Document any bugs or feature requests
- Provide relevant logs and error messages

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Changelog

### Version 1.0.0
- Initial release
- Basic file processing
- AI model integration
- Conversation history
- Export functionality

### Version 1.1.0
- Improved content chunking
- Enhanced error handling
- Added progress tracking
- Cache optimization