from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
# from langchain.schema import Document
from langchain_core.documents import Document
from pathlib import Path
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from prompt import MASTER_PROMPTT

def content_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()



### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

# Process all PDFs in the data directory
# all_pdf_documents = process_all_pdfs("../data")




def flatten_json_to_text(data, prefix=""):
    """
    Recursively flattens any JSON into path-aware text.
    """
    lines = []

    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            lines.extend(flatten_json_to_text(v, new_prefix))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            lines.extend(flatten_json_to_text(item, new_prefix))

    else:
        # leaf node
        if isinstance(data, (str, int, float, bool)):
            lines.append(f"{prefix}: {data}")

    return lines



def process_all_jsons(json_directory, text_fields=None):
    """
    Process all JSON files in a directory into LangChain Documents
    
    text_fields: list of keys to extract text from (None = auto-detect strings)
    """
    all_documents = []
    json_dir = Path(json_directory)
    json_files = list(json_dir.glob("**/*.json"))

    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Normalize to list
            records = data if isinstance(data, list) else [data]

            for idx, record in enumerate(records):
                if not isinstance(record, dict):
                    continue

                flattened_lines = flatten_json_to_text(record)
                content = "\n".join(flattened_lines)

                if not content.strip():
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        "source_file": json_file.name,
                        "file_type": "json",
                        "record_index": idx
                    }
                )
                all_documents.append(doc)

            print(f"  ✓ Loaded {len(records)} records")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal JSON documents loaded: {len(all_documents)}")
    return all_documents


# pdf_docs = process_all_pdfs("../data/pdfs")
# json_docs = process_all_jsons("../data/jsons", text_fields=["title", "content", "body"])

# all_documents = pdf_docs + json_docs


### Text splitting get into chunks

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs



# chunks=split_documents(all_documents)
# chunks



class EmbeddingManager:
    '''
    Handles document generation using sentence transformer
    '''
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model laoded successfully. Embedding dimesion: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        takes list of strings as input
        returns a numpy array of embeddings 
        """
        if not self.model:
            raise ValueError("Model not found")

        print (f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings


## initialise the embedding manager

# embedding_manager = EmbeddingManager()
# embedding_manager
         


class VectorStore:
    # manages document embeddings in chromadb vector store

    def __init__ (self, collection_name: str = "knowledge_base", persistent_directory: str = "../data/vector_store"):
        """
        Initialise the vector store
        collection_name - name of chormadb collection
        persistent_directory - directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._initialise_store()


    def _initialise_store(self):
        # initialise chromadb clinet and collection
        try:
            # create a persistant chromadb client
            os.makedirs(self.persistent_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persistent_directory)

            # get or create collectiopn
            self.collection = self.client.get_or_create_collection( 
                name=self.collection_name,
                metadata={
                "description": "PDF + JSON embeddings for RAG",
                "hnsw:space": "cosine"
                }
            )

            print(f"Vector Store initialised: {self.collection_name}")
            print(f"Existing documents in collectionL {self.collection.count()}")

        except Exception as e:
            print(f"Error initialising vector store {e}")
            raise

    def reset_collection(self):
        """Delete and recreate the current collection."""
        try:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF + JSON embeddings for RAG",
                    "hnsw:space": "cosine"
                }
            )
            print(f"Reset vector store collection: {self.collection_name}")
        except Exception as e:
            print(f"Error resetting vector store {e}")
            raise



    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """ 
        add documents and their embeddings to vector store
        Documents: list of langchain documents
        embeddings - coresponding embeddings for documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("no. of docx must match no. of embeddings")
        print(f"Adding {len(documents)} documents to vector store")

        # prepare data for chroma db
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list =[]

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # prepare metadata
            metadata = dict(doc.metadata)
            # metadata['doc_index'] = 1
            metadata['doc_index'] = i

            metadata['content_length'] = len(doc.page_content)
            metadata['content_hash'] = content_hash(doc.page_content)

            metadatas.append(metadata)

            #document content
            documents_text.append(doc.page_content)

            #embedding
            embeddings_list.append(embedding.tolist())
        
        # add to colection
        try: 
            self.collection.add(
                ids = ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents= documents_text
            )
            print(f"Successfully added {len(documents)} docx to vector store")
            print(f"Total docx in collection : {self.collection.count()}")

        except Exception as e:
            print(f"error adding docx to vector soter: {e}")
            raise

# vectorstore = VectorStore()
# vectorstore



# # convert text to embeddings
# texts = [doc.page_content for doc in chunks]

# # Generate the embeddings
# emeddings = embedding_manager.generate_embeddings(texts)

# # store in vector db
# vectorstore.add_documents(chunks, emeddings) 


class RAGRetriever:
    ''' handles query based retrival from vector store'''

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        ''' initialise the retirver, '''
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager


    def retrive(self, query: str, top_k: int=5, score_threshold: float=0.0) -> List[Dict[str, Any]]:
        ''' retirve relevant docx from query
            query - the search query
            topk - no. of top results to return
            score_threshold - minimum similarity score threshold
            returns - list of dictionaires containing retirved documents and metadata
        '''
        print(f"Retriving documents for query {query}")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        #generate query embeddings
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        ## Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

# rag_retriever=RAGRetriever(vectorstore,embedding_manager)
        


# load_dotenv()

# # initialise groq
# groq_api_key = os.getenv("GROQ_API_KEY")

# # init llm
# llm=ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-120b", temperature=0.1, max_tokens=1024)

# rag function

def rag_simple(query, retriever, llm, top_k=3):
    # retrive the context
    results = retriever.retrive(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context: 
        return "no relevant context found"
     
    # generate ans using prompt
    prompt=f"""Use the following context to answer the following question 
    Context: 
    {context}
    Question:
    {query}
    Answer: 
    """

    response=llm.invoke([prompt.format(context=context, query=query)])
    return response.content



# answer = rag_simple("who wrote the book", rag_retriever, llm)
# print(answer)

# --- Enhanced RAG Pipeline Features ---
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrive(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    
    # Prepare context and sources
    MASTER_PROMPT = MASTER_PROMPTT
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])
    
    # Generate answer
    prompt = f"""{MASTER_PROMPT}.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context, query=query)])
    
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output

def rag_advanced_v2(
    query: str,
    retriever,
    llm,
    *,
    retrieve_k: int = 12,
    final_k: int = 4,
    min_score: float = 0.25,
    max_context_chars: int = 3500,
    return_context: bool = False
):
    """
    Professional-grade RAG core.
    """

    # 1. Broad retrieval (recall phase)
    retrieved = retriever.retrive(
        query,
        top_k=retrieve_k,
        score_threshold=0.0  # do NOT filter early
    )

    if not retrieved:
        return {
            "answer": "I don't know. No relevant information was found.",
            "confidence": 0.0,
            "sources": [],
            "context": "" if return_context else None
        }

    # 2. Hard score filtering (precision gate)
    filtered = [
        d for d in retrieved
        if d["similarity_score"] >= min_score
    ]

    if not filtered:
        return {
            "answer": "I don't know. The available information is too weak to answer reliably.",
            "confidence": 0.0,
            "sources": [],
            "context": "" if return_context else None
        }

    # 3. Sort by relevance (descending)
    filtered.sort(key=lambda x: x["similarity_score"], reverse=True)

    # 4. Redundancy control (simple but effective)
    seen_hashes = set()
    selected = []

    for doc in filtered:
        h = doc["metadata"].get("content_hash")
        if h and h in seen_hashes:
            continue
        seen_hashes.add(h)
        selected.append(doc)
        if len(selected) >= final_k:
            break

    # 5. Context budgeting
    context_chunks = []
    total_chars = 0

    for doc in selected:
        chunk = doc["content"].strip()
        if total_chars + len(chunk) > max_context_chars:
            break
        context_chunks.append(chunk)
        total_chars += len(chunk)

    context = "\n\n---\n\n".join(context_chunks)

    # 6. Prompt (tight, enforceable contract)
    prompt = f"""
You are an expert assistant.

Rules:
- Use ONLY the provided context.
- If the context does not contain the answer, say: "I don't know."
- Be concise, precise, and factual.
- Do not speculate.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    response = llm.invoke(prompt)
    answer = response.content.strip()

    # 7. Confidence estimation (honest, not fake)
    confidence = max(d["similarity_score"] for d in selected)

    sources = [
        {
            "source": d["metadata"].get("source_file", "unknown"),
            "score": round(d["similarity_score"], 3),
            "preview": d["content"][:200] + "..."
        }
        for d in selected
    ]

    output = {
        "answer": answer,
        "confidence": round(confidence, 3),
        "sources": sources
    }

    if return_context:
        output["context"] = context

    return output



# # Example usage:
# result = rag_advanced("examples of confirmation bias", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
# print("Answer:", result['answer'])
# print("Sources:", result['sources'])
# print("Confidence:", result['confidence'])
# print("Context Preview:", result['context'][:300])
def build_rag(
    persist_dir="./data/vector_store",
    collection_name="knowledge_base"
):
    load_dotenv()

    # 1. Load embedder (for queries only)
    embedder = EmbeddingManager()

    # 2. Load vector store (NO ADDING DOCS)
    store = VectorStore(
        collection_name=collection_name,
        persistent_directory=persist_dir
    )

    # 3. Retriever
    retriever = RAGRetriever(store, embedder)

    # 4. LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1024
    )

    # 5. Callable chat function
    def ask(question, top_k=5, min_score=0.2, return_context=False):
        return rag_advanced(
            query=question,
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            min_score=min_score,
            return_context=return_context
        )

    return ask



def build_rag_v2(
    persist_dir: str = "./data/vector_store",
    collection_name: str = "knowledge_base",
    model_name: str = "openai/gpt-oss-120b",
):
    load_dotenv()

    # 1. Embedder (queries only)
    embedder = EmbeddingManager()

    # 2. Vector store (read-only)
    store = VectorStore(
        collection_name=collection_name,
        persistent_directory=persist_dir
    )

    # 3. Retriever
    retriever = RAGRetriever(store, embedder)

    # 4. LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=model_name,
        temperature=0.1,
        max_tokens=1024
    )

    def ask(
        question: str,
        *,
        retrieve_k: int = 12,
        final_k: int = 4,
        min_score: float = 0.25,
        max_context_chars: int = 3500,
        return_context: bool = False
    ):
        return rag_advanced_v2(
            query=question,
            retriever=retriever,
            llm=llm,
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_score=min_score,
            max_context_chars=max_context_chars,
            return_context=return_context
        )

    return ask
