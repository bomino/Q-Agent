# app.py
import streamlit as st
from openai import OpenAI, OpenAIError
from anthropic import Anthropic
import pandas as pd
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import tempfile
import os
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessor:
    """Handle different file types and their processing."""
    
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
    ALLOWED_TYPES = ["text/csv", "application/json", "text/plain"]
    ALLOWED_EXTENSIONS = ['.csv', '.json', '.txt']
    
    @staticmethod
    def validate_file(uploaded_file) -> tuple[bool, str]:
        """Validate uploaded file size and type."""
        if uploaded_file.size > FileProcessor.MAX_FILE_SIZE:
            return False, f"File {uploaded_file.name} is too large. Maximum size is 5MB."
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        if uploaded_file.type not in FileProcessor.ALLOWED_TYPES and file_extension not in FileProcessor.ALLOWED_EXTENSIONS:
            return False, f"File {uploaded_file.name} has unsupported type. Supported types are: CSV, JSON, and TXT."
        
        return True, "File is valid"
    
    @staticmethod
    def process_text_file(content: bytes) -> str:
        """Process text-based files."""
        return content.decode("utf-8")
    
    @staticmethod
    def process_csv(content: bytes) -> str:
        """Process CSV files into a structured string."""
        df = pd.read_csv(pd.io.common.BytesIO(content))
        return df.to_string()
    
    @staticmethod
    def process_json(content: bytes) -> str:
        """Process JSON files into a formatted string."""
        try:
            json_data = json.loads(content.decode("utf-8"))
            return json.dumps(json_data, indent=2)
        except json.JSONDecodeError:
            logger.error("Invalid JSON file")
            return "Error: Invalid JSON file"

class ResponseCache:
    """Simple cache for API responses."""
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=24)
    
    def _generate_key(self, content: str, query: str, model: str) -> str:
        """Generate a unique cache key."""
        combined = f"{content}:{query}:{model}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, content: str, query: str, model: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._generate_key(content, query, model)
        if key in self.cache:
            timestamp, response = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, content: str, query: str, model: str, response: str) -> None:
        """Cache a new response."""
        key = self._generate_key(content, query, model)
        self.cache[key] = (datetime.now(), response)
    
    def clear(self) -> None:
        """Clear expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (timestamp, _) in self.cache.items()
            if now - timestamp >= self.cache_duration
        ]
        for key in expired_keys:
            del self.cache[key]

def retry_with_backoff(retries: int = 3, backoff_factor: float = 0.5) -> Callable:
    """Decorator to retry API calls with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_count = 0
            while retry_count < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == retries:
                        raise e
                    
                    wait_time = backoff_factor * (2 ** (retry_count - 1))
                    logger.warning(f"Attempt {retry_count} failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class ContentProcessor:
    """Handle content processing and chunking."""
    
    MODEL_TOKENS = {
        "gpt-3.5-turbo": 8192,
        "gpt-4": 8192,
        "claude-3-opus-20240229": 200000
    }
    
    # More conservative token estimation
    TOKENS_PER_CHAR = 0.4
    
    # Reserve tokens for system message, user query, and response
    RESERVED_TOKENS = 1000
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate the number of tokens in a text."""
        return int(len(text) * ContentProcessor.TOKENS_PER_CHAR)
    
    @staticmethod
    def get_chunk_size(model: str) -> int:
        """Get appropriate chunk size for a model."""
        max_tokens = ContentProcessor.MODEL_TOKENS.get(model, 4000)
        # More conservative chunk size calculation
        available_tokens = max_tokens - ContentProcessor.RESERVED_TOKENS
        return int((available_tokens * 0.8) / ContentProcessor.TOKENS_PER_CHAR)
    
    @staticmethod
    def chunk_content(content: str, model: str = "gpt-3.5-turbo") -> List[str]:
        """Split content into chunks respecting token limits."""
        chunk_size = ContentProcessor.get_chunk_size(model)
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            estimated_tokens = ContentProcessor.estimate_tokens(paragraph)
            
            # If a single paragraph is too large, split it into sentences
            if paragraph_size > chunk_size:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence_size = len(sentence)
                    if current_size + sentence_size > chunk_size:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sentence_size
                    else:
                        current_chunk.append(sentence)
                        current_size += sentence_size
            # If adding the paragraph would exceed chunk size, start a new chunk
            elif current_size + paragraph_size > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


class AIModelInterface:
    """Interface for different AI models."""
    
    def __init__(self):
        self.cache = ResponseCache()
    
    @retry_with_backoff(retries=3)
    def call_openai(self, api_key: str, knowledge_base: str, user_query: str, model: str = "gpt-3.5-turbo") -> str:
        """Call OpenAI API with error handling and retry logic."""
        try:
            cached_response = self.cache.get(knowledge_base, user_query, model)
            if cached_response:
                return cached_response
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes provided knowledge base content."},
                    {"role": "user", "content": f"Knowledge Base:\n{knowledge_base}\n\nQuery: {user_query}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            self.cache.set(knowledge_base, user_query, model, result)
            return result
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise APIError(f"OpenAI API error: {str(e)}")
    
    @retry_with_backoff(retries=3)
    def call_anthropic(self, api_key: str, knowledge_base: str, user_query: str, model: str = "claude-3-opus-20240229") -> str:
        """Call Anthropic Claude API with error handling."""
        try:
            cached_response = self.cache.get(knowledge_base, user_query, model)
            if cached_response:
                return cached_response
            
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model,
                max_tokens=1000,
                system="You are a helpful assistant that analyzes provided knowledge base content.",
                messages=[{
                    "role": "user",
                    "content": f"Knowledge Base:\n{knowledge_base}\n\nQuery: {user_query}"
                }]
            )
            
            result = message.content[0].text if hasattr(message.content[0], 'text') else message.content[0].value
            
            self.cache.set(knowledge_base, user_query, model, result)
            return result
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise APIError(f"Anthropic API error: {str(e)}")

    def process_chunks(self, api_key: str, chunks: List[str], user_query: str, model_choice: str) -> str:
        """Process content chunks and combine results."""
        responses = []
        
        for i, chunk in enumerate(chunks):
            # Create a summarized version of the query for each chunk
            chunk_query = f"Analyze this portion of the content (Part {i+1}/{len(chunks)}):\n{user_query}"
            
            try:
                if "OpenAI" in model_choice:
                    model = "gpt-4" if model_choice == "OpenAI GPT-4" else "gpt-3.5-turbo"
                    response = self.call_openai(api_key, chunk, chunk_query, model)
                else:
                    response = self.call_anthropic(api_key, chunk, chunk_query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                responses.append(f"Error in chunk {i+1}: {str(e)}")
        
        # Create a summary prompt that's mindful of token limits
        summary_prompt = "Synthesize the following analysis results into a coherent summary. Focus on the key points and insights:"
        combined_response = f"{summary_prompt}\n\n" + "\n\n---\n\n".join(
            f"Analysis {i+1}:\n{response}" for i, response in enumerate(responses)
        )
        
        # Get final summary
        try:
            if "OpenAI" in model_choice:
                model = "gpt-4" if model_choice == "OpenAI GPT-4" else "gpt-3.5-turbo"
                final_response = self.call_openai(
                    api_key,
                    combined_response,
                    "Provide a concise, coherent summary of all the analyses.",
                    model
                )
            else:
                final_response = self.call_anthropic(
                    api_key,
                    combined_response,
                    "Provide a concise, coherent summary of all the analyses."
                )
            return final_response
        except Exception as e:
            logger.error(f"Error in final summary generation: {str(e)}")
            return "Error generating final summary. Please try again with a smaller content size or different query."


class SessionState:
    """Manage Streamlit session state."""
    
    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if 'knowledge_base' not in st.session_state:
            st.session_state.knowledge_base = ""
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'ai_interface' not in st.session_state:
            st.session_state.ai_interface = AIModelInterface()
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'query' not in st.session_state:
            st.session_state.query = ""
            
    @staticmethod
    def add_to_history(query: str, response: str):
        """Add a query-response pair to the conversation history."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversation_history.append({
            'timestamp': timestamp,
            'query': query,
            'response': response
        })

def render_sidebar():
    """Render the sidebar with configuration options and help sections."""
    with st.sidebar:
        # Help Sections
        with st.expander("ℹ️ About"):
            st.markdown("""
                ## Knowledge Base AI Assistant
                
                This application helps you analyze documents using advanced AI models.
                Built with Streamlit and powered by OpenAI and Anthropic.
                
                **Version:** 1.1.0
                
                **Features:**
                - Multiple file format support
                - AI-powered analysis
                - Conversation history
                - Export capabilities
            """)
        
        with st.expander("📖 How to Use"):
            st.markdown("""
                ### Quick Start Guide
                
                1. **Configuration**
                   - Enter your API key below
                   - Select your preferred AI model
                
                2. **Upload Files**
                   - Click 'Upload Knowledge Base Files'
                   - Supported formats: CSV, JSON, TXT
                   - Maximum file size: 5MB
                
                3. **Analysis**
                   - Type your query in the text area
                   - Click 'Analyze' to process
                   - View results below
                
                4. **Export**
                   - Download conversation history
                   - Export knowledge base
                
                ### Tips
                - Clear queries between analyses
                - Use specific questions
                - Check processing details in expandable sections
            """)
        
        st.markdown("---")
        
        # Existing configuration options
        st.header("Configuration")
        model_choice = st.selectbox(
            "Select AI Model:",
            options=["OpenAI GPT-3.5", "OpenAI GPT-4", "Claude AI"],
            help="Choose the AI model for analysis"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Controls randomness in responses"
        )
        
        st.header("Processed Files")
        if st.session_state.processed_files:
            for file in st.session_state.processed_files:
                st.text(f"✓ {file}")
        
        if st.button("Clear Cache"):
            st.session_state.ai_interface.cache.clear()
            st.success("Cache cleared successfully!")
        
        return model_choice, temperature

def process_file(uploaded_file) -> tuple[bool, str]:
    """Process uploaded file based on its type."""
    is_valid, message = FileProcessor.validate_file(uploaded_file)
    if not is_valid:
        return False, message
    
    try:
        file_content = uploaded_file.read()
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if uploaded_file.type == "text/csv" or file_extension == '.csv':
            content = FileProcessor.process_csv(file_content)
        elif uploaded_file.type == "application/json" or file_extension == '.json':
            content = FileProcessor.process_json(file_content)
        else:
            content = FileProcessor.process_text_file(file_content)
        
        return True, content
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return False, f"Error processing file: {str(e)}"

def clear_query():
    """Callback function to clear the query."""
    st.session_state.query = ""

def clear_all():
    """Callback function to clear all data."""
    st.session_state.knowledge_base = ""
    st.session_state.processed_files = []
    st.session_state.conversation_history = []
    st.session_state.query = ""

def main():
    st.set_page_config(
        page_title="Knowledge Base AI Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    SessionState.initialize()
    
    # Apply custom CSS
    st.markdown("""
        <style>
        .main > div {
            padding: 2rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .section-divider {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        .custom-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with Logo
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("QLogo.png", width=220)
    with col2:
        st.title("📚 Knowledge Base AI Assistant")
    st.markdown("---")
    
    # Sidebar configuration
    model_choice, temperature = render_sidebar()
    
    # Parameters Section
    st.header("⚙️ Configuration")
    with st.container():
        with st.expander("API Configuration", expanded=True):
            api_key = st.text_input("Enter your API Key:", type="password")
            if api_key:
                st.success("✅ API Key set successfully!")
        
        with st.expander("Knowledge Base Files", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload Knowledge Base Files",
                accept_multiple_files=True,
                type=["txt", "csv", "json"],
                help="Supported formats: CSV, JSON, and TXT files"
            )
            
            if uploaded_files:
                with st.spinner("Processing files..."):
                    progress_bar = st.progress(0)
                    for i, uploaded_file in enumerate(uploaded_files):
                        success, result = process_file(uploaded_file)
                        if success:
                            st.session_state.knowledge_base += f"\n### {uploaded_file.name} ###\n{result}\n"
                            st.session_state.processed_files.append(uploaded_file.name)
                        else:
                            st.error(result)
                        progress_bar.progress((i + 1) / len(uploaded_files))

                        st.success(f"✅ Processed {len(uploaded_files)} files successfully!")
    
    st.markdown("---")
    
    # Analysis Section
    st.header("🔍 Analysis")
    with st.container():
        # Query input with callback functions
        user_query = st.text_area(
            "Enter your query:",
            height=100,
            key="query",
            value=st.session_state.query,
            placeholder="What would you like to know about your knowledge base?"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
        with col2:
            clear_query_button = st.button("🔄 Clear Query", on_click=clear_query, use_container_width=True)
        with col3:
            clear_all_button = st.button("🗑️ Clear All", on_click=clear_all, use_container_width=True)

        # In your main function, replace the analysis section with:
        if analyze_button and user_query:
            if not all([api_key, st.session_state.knowledge_base, user_query]):
                st.error("⚠️ Please ensure all required fields are filled.")
            else:
                with st.spinner("🔄 Analyzing..."):
                    try:
                        # Always chunk the content to ensure consistent handling
                        chunks = ContentProcessor.chunk_content(
                            st.session_state.knowledge_base,
                            "gpt-3.5-turbo" if "OpenAI GPT-3.5" in model_choice else "gpt-4"
                        )
                        
                        with st.expander("Processing Details", expanded=False):
                            st.caption(f"Content split into {len(chunks)} chunks for processing")
                            progress_bar = st.progress(0)
                            
                            final_response = st.session_state.ai_interface.process_chunks(
                                api_key, chunks, user_query, model_choice
                            )
                            progress_bar.progress(1.0)
                        
                        st.markdown("### 📊 Analysis Results")
                        with st.container():
                            st.markdown(final_response)
                        
                        SessionState.add_to_history(user_query, final_response)
                    
                    except APIError as e:
                        st.error(f"⚠️ {str(e)}")
                    except Exception as e:
                        st.error(f"⚠️ Error during analysis: {str(e)}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("### 💬 Conversation History")
            
            if st.download_button(
                label="📥 Export Conversation History",
                data=json.dumps(st.session_state.conversation_history, indent=2),
                file_name="conversation_history.json",
                mime="application/json",
                use_container_width=True
            ):
                st.success("✅ Conversation history exported successfully!")
            
            for i, entry in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q&A #{len(st.session_state.conversation_history) - i}", expanded=(i == 0)):
                    st.markdown(f"**🕒 {entry['timestamp']}**")
                    st.markdown("**🤔 Question:**")
                    st.markdown(entry['query'])
                    st.markdown("**💡 Answer:**")
                    st.markdown(entry['response'])
        
        # Export button for knowledge base
        if st.session_state.knowledge_base:
            st.markdown("---")
            st.markdown("### 💾 Export")
            if st.download_button(
                label="📥 Export Knowledge Base",
                data=st.session_state.knowledge_base,
                file_name="knowledge_base.txt",
                mime="text/plain",
                use_container_width=True
            ):
                st.success("✅ Knowledge base exported successfully!")

if __name__ == "__main__":
    main()