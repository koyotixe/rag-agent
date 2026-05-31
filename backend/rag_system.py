import os
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_gigachat.chat_models import GigaChat

try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

RAG_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "rag_documents")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "document_index")

if not os.path.exists(RAG_DOCUMENTS_DIR):
    os.makedirs(RAG_DOCUMENTS_DIR)

GIGACHAT_CREDENTIALS = "MDE5Yzc2MmUtYWFjYi03NDE4LTgyYTUtM2QxN2VlYTQ5NmU2OjU0MzNkZjk4LWRkOGUtNDNjMS04OTNkLTIxNjU5MzdjZjg2Ng=="

_llm = None
_vectorstore = None
_embeddings = None
executor = ThreadPoolExecutor(max_workers=1)

def get_llm():
    global _llm
    if _llm is None:
        _llm = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            model="GigaChat",
            temperature=0.7,
            max_tokens=512,
        )
    return _llm

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

def get_loader_for_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        try:
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(file_path)
        except ImportError:
            raise Exception("For PDF support install: pip install pypdf")
    else:
        return TextLoader(file_path, encoding='utf-8')

def load_all_documents():
    global _vectorstore
    
    files = []
    for file in os.listdir(RAG_DOCUMENTS_DIR):
        file_path = os.path.join(RAG_DOCUMENTS_DIR, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.txt', '.pdf']:
                files.append(file_path)
    
    if not files:
        return False
    
    all_documents = []
    
    for file_path in files:
        try:
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source'] = os.path.basename(file_path)
            
            all_documents.extend(documents)
        except Exception as e:
            pass
    
    if not all_documents:
        return False
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "!", "?", ";", " ", ""]
    )
    docs = text_splitter.split_documents(all_documents)
    
    embeddings = get_embeddings()
    _vectorstore = FAISS.from_documents(docs, embeddings)
    _vectorstore.save_local(INDEX_PATH)
    
    return True

def load_document(file_path: str) -> bool:
    global _vectorstore
    
    if not os.path.isabs(file_path) and not os.path.exists(file_path):
        file_path = os.path.join(RAG_DOCUMENTS_DIR, file_path)
    
    if not os.path.exists(file_path):
        return False
    
    try:
        loader = get_loader_for_file(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "!", "?", ";", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        
        embeddings = get_embeddings()
        _vectorstore = FAISS.from_documents(docs, embeddings)
        _vectorstore.save_local(INDEX_PATH)
        
        return True
    except Exception as e:
        return False

def load_existing_index() -> bool:
    global _vectorstore
    
    if os.path.exists(INDEX_PATH):
        try:
            embeddings = get_embeddings()
            _vectorstore = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            pass
    
    return load_all_documents()

def reload_all_documents():
    if os.path.exists(INDEX_PATH):
        import shutil
        shutil.rmtree(INDEX_PATH)
    return load_all_documents()

async def get_rag_answer(question: str) -> str:
    global _vectorstore
    
    if _vectorstore is None:
        return "Документ не загружен. Положите файлы в папку backend/rag_documents и перезапустите сервер."
    
    try:
        docs = _vectorstore.max_marginal_relevance_search(question, k=8, fetch_k=30)
        
        if not docs:
            return "Не нашел информации по вашему вопросу в лекциях."
        
        context_parts = []
        for i, doc in enumerate(docs[:6]):
            source = doc.metadata.get('source', 'неизвестный источник')
            context_parts.append(f"[Из {source}]:\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Ты - ассистент, который отвечает на вопросы по лекциям. Используй ТОЛЬКО информацию из контекста.

КОНТЕКСТ (из лекций):
{context}

ВОПРОС: {question}

ОТВЕТ (кратко, только на основе контекста):"""
        
        llm = get_llm()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: llm.invoke(prompt)
        )
        
        return response.content
        
    except Exception as e:
        return f"Ошибка при поиске ответа: {e}"

def is_rag_ready() -> bool:
    return _vectorstore is not None

def get_documents_list() -> list:
    if not os.path.exists(RAG_DOCUMENTS_DIR):
        return []
    
    files = []
    for file in os.listdir(RAG_DOCUMENTS_DIR):
        file_path = os.path.join(RAG_DOCUMENTS_DIR, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.txt', '.pdf']:
                files.append(file)
    return files