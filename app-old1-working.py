import os
import re
import hashlib
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLM and embedding imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Vector DB imports
from langchain_chroma import Chroma

# Memory imports
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Text to speech imports
from gtts import gTTS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
KB_FOLDER = "KB"  # Folder containing knowledge base files
METADATA_FILE = "kb_metadata.json"  # File to store metadata about processed files
PERSIST_DIRECTORY = "chroma_db"  # Directory for Chroma DB
CHAT_HISTORY_DIR = "chat_logs"  # Directory to store chat history

class KnowledgeBase:
    """Class to handle knowledge base operations"""
    
    def __init__(self, embedding_model):
        """Initialize the knowledge base"""
        self.embedding_model = embedding_model
        try:
            os.makedirs(KB_FOLDER, exist_ok=True)
            os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        except OSError as e:
            print("Error creating directories : {e}")

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
        
        # If we get here, we either have no documents or couldn't create a vector db
        raise ValueError("Could not create or load vector database")

class NexalyzeChatbot:
    """Main chatbot class"""
    
    def __init__(self):
        """Initialize the chatbot"""
        # Initialize LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
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
            
            If asked about CEO of nexalyze, answer with : "Muhammad Mubashar is the CEO of Nexalyze"
            If asked about CTO of nexalyze, answer with : "Abdul Rehman Siddique is the CTO of Nexalyze"
            If the context doesn't provide enough information or the question is unrelated to Nexalyze,
            Politely inform the user that the question is not related to Nexalyze and provide a general response based on your knowledge.
            Keep your responses upto 100 words.
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
        
        return response_text
    
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
            match = re.match(r"session_(\d+)\.txt", file)
            if match:
                session_numbers.append(int(match.group(1)))
        
        return max(session_numbers, default=0) + 1
    
    def save_chat_history(self, session_id: str):
        """Save chat history to a file"""
        if session_id not in self.memory_store:
            return
        
        message_history = self.memory_store[session_id]
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            for msg in message_history.messages:
                role = "You" if msg.type == "human" else "Nexalyze AI"
                f.write(f"{role}: {msg.content}\n")
        
        print(f"Chat history saved as {session_id}.txt")
    
    def run_cli(self):
        """Run the CLI interface for the chatbot"""
        session_index = self.get_next_session_index()
        session_id = f"session_{session_index}"
        
        print("Hi I am Marcus, Nexalyze AI Assistant: How can I help you today?")
        print("-" * 50)
        
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
        chatbot = NexalyzeChatbot()
        chatbot.run_cli()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()