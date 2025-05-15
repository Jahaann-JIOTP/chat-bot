#!/usr/bin/env python
# Set USER_AGENT environment variable before anything else
import os
import sys
###
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import concurrent.futures
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
from fake_useragent import UserAgent
from fpdf import FPDF
import io
from langchain_community.document_loaders import WebBaseLoader
###

# Set USER_AGENT environment variable and verify it's set
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Nexalyze/1.0"

# Verify it's set to debug any issues
print(f"USER_AGENT is set to: {os.environ.get('USER_AGENT')}")

# Rest of the imports
import re
import hashlib
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging


PERSIST_DIRECTORY = "/var/www/html/chat-bot/vector_store"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraper_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger("WebScraper")

class AdvancedWebScraper:
    def __init__(self, max_depth=2, delay_range=(1, 3), max_workers=5, timeout=30):
        self.max_depth = max_depth
        self.delay_range = delay_range
        self.max_workers = max_workers
        self.timeout = timeout
        # Initialize user agent with fallback
        try:
            self.ua = UserAgent()
            # Test if it works
            test_agent = self.ua.random
            logger.info(f"UserAgent initialized successfully: {test_agent}")
        except Exception as e:
            logger.warning(f"Error initializing UserAgent: {e}. Using fallback.")
            # Fallback to a standard user agent
            self.ua = None
        self.visited_urls = set()
        self.results = []
        # Define a fallback user agent
        self.fallback_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Nexalyze/1.0"

    def get_random_headers(self):
        try:
            user_agent = self.ua.random if self.ua else self.fallback_user_agent
        except Exception:
            user_agent = self.fallback_user_agent
            
        logger.info(f"Using User-Agent: {user_agent}")
        
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }

    def fetch_url(self, url, depth=3):
        if url in self.visited_urls or depth > self.max_depth:
            return None

        self.visited_urls.add(url)
        time.sleep(random.uniform(*self.delay_range))
        logger.info(f"Fetching: {url} (Depth: {depth})")

        try:
            response = requests.get(
                url,
                headers=self.get_random_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            domain = urlparse(url).netloc
            base_url = f"{urlparse(url).scheme}://{domain}"
            title = soup.title.text.strip() if soup.title else "No title"
            text_content = self.extract_text(soup)
            tables = self.extract_tables(soup)
            links = []
            if depth < self.max_depth:
                links = self.extract_links(soup, base_url)
            images = self.extract_images(soup, base_url)
            metadata = self.extract_metadata(soup)

            result = {
                'url': url,
                'title': title,
                'text': text_content,
                'tables': tables,
                'links': links,
                'images': images,
                'metadata': metadata,
                'depth': depth
            }

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_text(self, soup):
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()

        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = self.clean_text('\n'.join(lines))
        return cleaned_text

    def clean_text(self, text):
        replacements = {
            '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '--',
            '\u2026': '...',
            '\u00a0': ' ',
            '\u00ad': '',
            '\u2022': '*',
            '\u2212': '-',
            '\u00b7': '*',
            '\u00b0': 'deg',
            '\u00ae': '(R)',
            '\u00a9': '(C)',
            '\u00e9': 'e',
            '\u00e8': 'e',
            '\u00ea': 'e',
            '\u00eb': 'e',
            '\u00e0': 'a',
            '\u00e1': 'a',
            '\u00e2': 'a',
            '\u00e4': 'a',
            '\u00e7': 'c',
            '\u00ee': 'i',
            '\u00ef': 'i',
            '\u00f4': 'o',
            '\u00f6': 'o',
            '\u00fb': 'u',
            '\u00fc': 'u',
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        cleaned_text = ''
        for char in text:
            if ord(char) < 128:
                cleaned_text += char
            else:
                cleaned_text += '?'

        return cleaned_text

    def extract_tables(self, soup):
        tables = []

        for table in soup.find_all('table'):
            try:
                df = pd.read_html(str(table))[0]
                table_data = df.to_dict('records')
                cleaned_table = []

                for row in table_data:
                    cleaned_row = {}
                    for key, value in row.items():
                        cleaned_key = self.clean_text(str(key))
                        cleaned_value = self.clean_text(str(value))
                        cleaned_row[cleaned_key] = cleaned_value
                    cleaned_table.append(cleaned_row)

                tables.append(cleaned_table)
            except Exception as e:
                logger.warning(f"Error processing table: {str(e)}")
                continue

        return tables

    def extract_links(self, soup, base_url):
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('#') or href.startswith('javascript:'):
                continue
            if not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            if urlparse(base_url).netloc == urlparse(href).netloc:
                links.append(href)

        return list(set(links))

    def extract_images(self, soup, base_url):
        images = []

        for img in soup.find_all('img', src=True):
            src = img['src']
            if not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            alt_text = img.get('alt', '')
            alt_text = self.clean_text(alt_text)
            images.append({'src': src, 'alt': alt_text})

        return images

    def extract_metadata(self, soup):
        metadata = {}

        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                name = self.clean_text(str(name))
                content = self.clean_text(str(content))
                metadata[name] = content

        return metadata

    def scrape_urls(self, urls):
        self.results = []
        self.visited_urls = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            initial_futures = {executor.submit(self.fetch_url, url): url for url in urls}
            for future in tqdm(concurrent.futures.as_completed(initial_futures), total=len(initial_futures), desc="Processing initial URLs"):
                result = future.result()
                if result and result['links']:
                    pass  # Handle recursive crawling if needed here

# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM and embedding imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Vector DB imports
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma


# Memory imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Import our SafeWebLoader instead of using WebBaseLoader directly
from safe_web_loader import SafeWebLoader

# Text to speech imports
from gtts import gTTS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
KB_FOLDER = "KB"  # Folder containing knowledge base files
METADATA_FILE = "kb_metadata.json"  # File to store metadata about processed files
# PERSIST_DIRECTORY = "chroma_db"  # Directory for Chroma DB
CHAT_HISTORY_DIR = "chat_logs"  # Directory to store chat history


class KnowledgeBase:
    """Class to handle knowledge base operations"""
    
    def __init__(self, embedding_model):
        """Initialize the knowledge base"""
        self.embedding_model = embedding_model
        self.web_scraper = AdvancedWebScraper()
        # Define a fallback user agent for WebBaseLoader
        self.fallback_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Nexalyze/1.0"
        try:
            os.makedirs(KB_FOLDER, exist_ok=True)
            os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        except OSError as e:
            print("Error creating directories : {e}")
            
    def load_web_page(self, url):
        """Load content from a web page using SafeWebLoader with a proper user agent"""
        try:
            logger.info(f"Loading web page with SafeWebLoader: {url}")
            # Use SafeWebLoader which already handles user agent properly
            loader = SafeWebLoader(
                web_paths=[url],
                verify_ssl=True,
                requests_per_second=1  # Rate limiting to be respectful
            )
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} documents from {url}")
            return docs
        except Exception as e:
            logger.error(f"Error loading web page {url}: {e}")
            return []

    def save_web_content(self, links_list, filename="nexalyze_kb.txt", use_web_loader=False):
        """Save web content from a list of links to a text file in the KB folder
        
        Args:
            links_list: List of URLs to scrape
            filename: Output filename
            use_web_loader: If True, use WebBaseLoader instead of AdvancedWebScraper
        """
        output_path = os.path.join(KB_FOLDER, filename)
        try:
            web_data = ""
            
            if use_web_loader:
                logger.info("Using WebBaseLoader to fetch content")
                for url in links_list:
                    docs = self.load_web_page(url)
                    if docs:
                        web_data += f"URL: {url}\n"
                        for doc in docs:
                            web_data += f"Title: {doc.metadata.get('title', 'No title')}\n"
                            web_data += f"Text Content:\n{doc.page_content}\n\n"
                        web_data += "-" * 80 + "\n\n"
            else:
                logger.info("Using AdvancedWebScraper to fetch content")
                # Scrape the URLs using AdvancedWebScraper
                self.web_scraper.scrape_urls(links_list)
                
                # Combine all scraped content
                for result in self.web_scraper.results:
                    web_data += f"URL: {result['url']}\n"
                    web_data += f"Title: {result['title']}\n"
                    web_data += f"Text Content:\n{result['text']}\n\n"
                    
                    if result['tables']:
                        web_data += "Tables:\n"
                        for table in result['tables']:
                            web_data += str(table) + "\n\n"
                    
                    if result['metadata']:
                        web_data += "Metadata:\n"
                        for key, value in result['metadata'].items():
                            web_data += f"{key}: {value}\n"
                    
                    web_data += "-" * 80 + "\n\n"

            # Clean and save the content
            web_data = re.sub(r'\n+', '\n', web_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(web_data)
            logger.info(f"Web content saved to {output_path}")
        except Exception as e:
            logger.error(f"Error scraping data from websites: {e}")
        return output_path

    def get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file to detect changes"""
        with open(file_path, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()
    
    def load_text_file(self, file_path: str) -> str:
        """Load text from a file"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
        except FileNotFoundError:
            print(f"error reading {file_path} file")
            return ""
        except Exception as e:
            print(f'Unexpected Error : {e}')
            return ""

    def load_pdf_file(self, file_path: str) -> str:
        """Load text from a PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
        except FileNotFoundError:
            print(f"error reading {file_path} file")
            return ""
        except Exception as e:
            print(f'Unexpected Error : {e}')
            return ""
        
    def chunk_text(self, text: str, chunk_size: int = 750, chunk_overlap: int = 120) -> List[str]:
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)
    
    def scan_kb_folder(self) -> Dict[str, str]:
        """Scan the KB folder for text and PDF files and their hashes"""
        file_hashes = {}
        for filename in os.listdir(KB_FOLDER):
            file_path = os.path.join(KB_FOLDER, filename)
            if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.pdf')):
                file_hashes[file_path] = self.get_file_hash(file_path)
        return file_hashes
    
    def load_metadata(self) -> Dict[str, str]:
        """Load metadata of previously processed files"""
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r') as file:
                    return json.load(file)
            except Exception as e:
                print(f'Error occured while loading metadata from json file : {e}')
                return {}
        return {}
    
    def save_metadata(self, metadata: Dict[str, str]):
        """Save metadata of processed files"""
        try:
            with open(METADATA_FILE, 'w') as file:
                json.dump(metadata, file, indent=4)
        except Exception as e:
            print(f'Error ocurred while writing metadata file : {e}')

    def process_kb_files(self) -> Tuple[List[str], bool]:
        """Process knowledge base files and determine if reindexing is needed"""
        current_files = self.scan_kb_folder()
        previous_files = self.load_metadata()
        
        # Check if any files are new or modified

        need_reindex = False
        # Check for any missing files 
        missing_files = set(previous_files.keys()) - set(current_files.keys())
        if missing_files:
            need_reindex = True
        if set(current_files.keys()) != set(previous_files.keys()):
            need_reindex = True
        else:
            for file_path, file_hash in current_files.items():
                if previous_files.get(file_path) != file_hash:
                    need_reindex = True
                    break
        
        if not need_reindex and previous_files:
            print("No changes detected in knowledge base files. Using existing vector database.")
            return [], False
        
        # Process all files if reindexing is needed
        all_chunks = []
        for file_path in current_files.keys():
            try:
                if file_path.endswith('.txt'):
                    text = self.load_text_file(file_path)
                elif file_path.endswith('.pdf'):
                    text = self.load_pdf_file(file_path)
                else:
                    continue
                
                chunks = self.chunk_text(text)
                all_chunks.extend(chunks)
                print(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save new metadata
        self.save_metadata(current_files)
        
        return all_chunks, True
    
    def get_or_create_vector_db(self, collection_name: str = "nexalyze_kb"):
        """Get existing vector DB or create a new one if needed"""
        documents, need_reindex = self.process_kb_files()
        
        if not need_reindex:
            # Try to load existing vector database
            try:
                vectordb = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=PERSIST_DIRECTORY
                )
                print(f"Loaded existing vector database with {vectordb._collection.count()} documents")
                return vectordb
            except Exception as e:
                print(f"Error loading existing vector database: {e}")
                need_reindex = True
        
        if need_reindex and documents:
            print(f"Creating new vector database with {len(documents)} chunks...")
            vectordb = Chroma.from_texts(
                texts=documents,
                embedding=self.embedding_model,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=PERSIST_DIRECTORY
            )
            print(f"Vector database ready with {vectordb._collection.count()} documents")
            return vectordb
        
        # If we get here, create an empty vector database
        print("No documents found in KB folder. Creating empty vector database...")
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        return vectordb


class NexalyzeChatbot:
    """Main chatbot class"""
    
    def __init__(self):
        """Initialize the chatbot"""
        # Initialize LLM
        self.llm = ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.2,
            max_tokens=1024
        )
        
        # Initialize embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        )
        
        # Initialize knowledge base
        self.kb = KnowledgeBase(self.embedding_model)
        self.vectordb = self.kb.get_or_create_vector_db()
        
        # Create a retriever
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 10,
            }
        )
        
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI assistant for Nexalyze, an AI technology company, you will be provided with most relevant information from company database, each time user asks a question.
            Use the following context to answer the user's question.

            Context: {context}

            If the question is about Nexalyze, use the provided context to answer.
            Stay strictly relevant to context.
            

            If the context doesn't provide enough information or the question is unrelated to Nexalyze,
            Politely inform the user that the question is not related to Nexalyze and provide a general response based on your knowledge.
            Keep your responses concise, professional and upto 100 words.
            Stay polite and professional.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Initialize memory store
        self.memory_store: Dict[str, BaseChatMessageHistory] = {}
    
    def get_message_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create message history for a session"""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ChatMessageHistory()
        return self.memory_store[session_id]
    
    def format_docs(self, docs):
        """Format retrieved documents into a context string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def chat(self, query: str, session_id: str = "default") -> str:
        """Chat with the Nexalyze bot"""
        # Get or create message history for this session
        message_history = self.get_message_history(session_id)
        
        # Retrieve context from vector store
        docs = self.retriever.invoke(query)
        context = self.format_docs(docs)
        
        # Create messages list from history
        history_messages = message_history.messages
        
        # Run the chain
        response = self.prompt_template.invoke({
            "context": context,
            "question": query,
            "history": history_messages
        })
        
        response_message = self.llm.invoke(response)
        response_text = response_message.content
        
        # Save the interaction to history
        message_history.add_user_message(query)
        message_history.add_ai_message(response_text)
        
        # Clean up memory if needed
        self.manage_memory()
        
        return response_text
    
    def manage_memory(self, max_sessions=100):
        """Manage memory by cleaning up old sessions if there are too many"""
        if len(self.memory_store) <= max_sessions:
            return
        
        # If there are too many sessions, remove the oldest ones
        # Sort sessions by last message time
        session_times = {}
        for session_id, history in self.memory_store.items():
            if history.messages:
                # Use the time of the last message as the session time
                session_times[session_id] = history.messages[-1].additional_kwargs.get(
                    'created_at', datetime.min
                )
            else:
                # If no messages, use the minimum date
                session_times[session_id] = datetime.min
        
        # Sort sessions by time (oldest first)
        sorted_sessions = sorted(session_times.items(), key=lambda x: x[1])
        
        # Keep only the most recent sessions up to max_sessions
        sessions_to_remove = sorted_sessions[:-max_sessions]
        
        for session_id, _ in sessions_to_remove:
            # Save the history before removing
            self.save_chat_history(session_id)
            # Remove from memory
            del self.memory_store[session_id]
            print(f"Removed old session {session_id} from memory (cleanup)")
            
        print(f"Memory cleanup complete. Active sessions: {len(self.memory_store)}")
    
    def text_to_speech(self, text: str, output_file: str = "response.mp3"):
        """Convert text to speech"""
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        return output_file
    
    def get_next_session_index(self):
        """Get the next session index for chat logging"""
        files = os.listdir(CHAT_HISTORY_DIR)
        session_numbers = []
        
        for file in files:
            match = re.match(r"session_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.txt", file)
            if match:
                session_numbers.append(match.group(1))
        
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def save_chat_history(self, session_id: str):
        """Save chat history to a file"""
        if session_id not in self.memory_store:
            return
        
        message_history = self.memory_store[session_id]
        
        # Skip empty sessions
        if not message_history.messages:
            return
            
        # Use the session_id for the filename to ensure each session has its own file
        # Clean the session_id to make it safe for a filename
        safe_session_id = re.sub(r'[^\w\-_]', '_', session_id)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{safe_session_id}_{timestamp}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Date: {timestamp}\n")
            f.write("-" * 50 + "\n\n")
            
            for msg in message_history.messages:
                role = "User" if msg.type == "human" else "Nexalyze AI"
                f.write(f"{role}: {msg.content}\n\n")
        
        print(f"Chat history saved for session {session_id}")
        return file_path
    
    def run_cli(self):
        """Run the CLI interface for the chatbot"""
        session_index = self.get_next_session_index()
        session_id = f"session_{session_index}"
        
        print("Hi I am Marcus, Nexalyze AI Assistant: How can I help you today?")
        print("-" * 50)
        
        # Check if KB folder is empty
        if not os.listdir(KB_FOLDER):
            print("No knowledge base files found in the KB folder.")
            scrape_web = input("Would you like to scrape data from websites to build your knowledge base? (yes/no): ").lower()
            if scrape_web in ["yes", "y"]:
                links_list = []
                print("Enter website URLs (enter 'done' when finished):")
                while True:
                    link = input("Enter URL: ")
                    if link.lower() == 'done':
                        break
                    links_list.append(link)
                
                if links_list:
                    print("Scraping web content...")
                    
                    # Ask user which scraper to use
                    scraper_choice = input("Choose a scraper method (1: AdvancedWebScraper, 2: WebBaseLoader): ")
                    use_web_loader = scraper_choice == "2"
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"web_content_{timestamp}.txt"
                    self.kb.save_web_content(links_list, filename, use_web_loader=use_web_loader)
                    print("Web content has been saved to the KB folder.")
                    # Refresh the vector database
                    self.vectordb = self.kb.get_or_create_vector_db()
            else:
                print("No problem! You can start chatting right away. If you want to add knowledge base files later, just place them in the KB folder.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Nexalyze AI: Goodbye!")
                self.save_chat_history(session_id)
                break
            
            response = self.chat(user_input, session_id)
            print(f"Nexalyze AI: {response}")
            print("-" * 50)


def main():
    """Main function to run the chatbot"""
    try:
        # Apply monkey patch to requests to always include User-Agent
        original_request = requests.Session.request
        
        def patched_request(self, method, url, **kwargs):
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            if 'User-Agent' not in kwargs['headers']:
                kwargs['headers']['User-Agent'] = os.environ.get('USER_AGENT', 
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Nexalyze/1.0")
            return original_request(self, method, url, **kwargs)
        
        requests.Session.request = patched_request
        
        chatbot = NexalyzeChatbot()
        chatbot.run_cli()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()