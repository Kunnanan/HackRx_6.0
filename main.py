# main.py (Version 3.0 - The Definitive Solution)

import os
import requests
import tempfile
import logging
import asyncio
from typing import List
from urllib.parse import urlparse
from dotenv import load_dotenv

from fastapi import FastAPI, Security, HTTPException, status, APIRouter
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SubmissionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

class QuerySystem:
    def __init__(self, document_url: str):
        self.document_url = document_url
        self.chain = self._initialize_pipeline()

    def _initialize_pipeline(self):
        try:
            logging.info("Initializing RAG pipeline...")
            docs = self._load_document_from_url()
            if not docs: raise ValueError("Document is empty.")
            chunks = self._split_document(docs)
            vector_store = self._create_vector_store(chunks)
            rag_chain = self._create_rag_chain(vector_store)
            logging.info("RAG pipeline initialized successfully.")
            return rag_chain
        except Exception as e:
            logging.error(f"Failed to initialize RAG pipeline: {e}")
            raise RuntimeError(f"Could not initialize the query system. Error: {e}")

    def _load_document_from_url(self):
        parsed_url = urlparse(str(self.document_url))
        file_ext = os.path.splitext(parsed_url.path)[1].lower()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        tmp_file_path = tmp_file.name
        try:
            with requests.get(self.document_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            tmp_file.close()
            loader_map = {'.pdf': PyPDFLoader, '.docx': Docx2txtLoader}
            loader = loader_map.get(file_ext)
            if not loader: raise ValueError(f"Unsupported file type: {file_ext}")
            return loader(tmp_file_path).load()
        finally:
            if os.path.exists(tmp_file_path): os.remove(tmp_file_path)

    def _split_document(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        return text_splitter.split_documents(docs)

    def _create_vector_store(self, chunks):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        return FAISS.from_documents(chunks, embeddings)

    # ========================================================================
    # =========== THE FINAL, NUANCED PROMPT FOR PERFECT OUTPUT ===============
    # ========================================================================
    def _create_rag_chain(self, vector_store):
        llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.1, max_tokens=500)
        
        # This final prompt provides nuanced rules to achieve the exact target output.
        prompt_template = """
**ROLE & GOAL:** You are a senior policy analyst creating a final, client-ready report. Your writing must be incredibly concise, direct, and authoritative.

**Golden Rules for Your Response:**

1.  **Direct Answer First (The MOST IMPORTANT Rule):**
    *   For questions starting with "What is...", "How does...", "What is the extent...", your answer MUST start directly with the information. Example: "A grace period of thirty days is provided...".
    *   For questions starting with "Does...", "Is...", "Are...", your answer MUST start with "Yes," or "No," followed by the explanation.

2.  **Summarize, Don't Enumerate (CRITICAL for Conciseness):**
    *   Do NOT list out every single exclusion or condition in a long list.
    *   Your primary skill is **synthesis**. Identify the 2-3 most critical conditions and weave them into a single, concise sentence or two.
    *   **PERFECT EXAMPLE:** For a maternity question, instead of listing all 7 exclusions, a perfect answer is: "Yes, the policy covers maternity expenses... To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period." This is the level of summary required.

3.  **Find the Exact Numbers:** The context contains specific percentages (e.g., "1% of Sum Insured", "5% NCD") and durations ("two (2) years"). Your response is only correct if it includes these exact figures. Hunt for them in the context and integrate them seamlessly.

4.  **No Conversational Filler:** Absolutely no phrases like "According to the policy...", "The policy states...".

**CONTEXT FROM DOCUMENT:**
{context}

**CLIENT QUESTION:**
{input}

**FINAL REPORT ENTRY:**
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever(search_kwargs={'k': 8}) # Increased k slightly for max context
        return create_retrieval_chain(retriever, question_answer_chain)
    # ========================================================================
    # ======================== END OF DEFINITIVE PROMPT ======================
    # ========================================================================


# --- FastAPI Application ---
app = FastAPI(title="Intelligent Query–Retrieval System (Definitive)", version="3.0.0")
API_KEY = os.getenv("HACKRX_TEAM_TOKEN")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_token(auth_header: str = Security(api_key_header)):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header.")
    token = auth_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Invalid token.")
    return token

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=SubmissionResponse,
    tags=["Submission"],
    summary="Process a document and answer questions with perfected output"
)
async def run_submission(request: SubmissionRequest, token: str = Security(get_token)):
    try:
        logging.info("Authentication successful.")
        if not os.getenv("GROQ_API_KEY"):
            raise HTTPException(status_code=500, detail="Server config error: GROQ_API_KEY not set.")
        
        query_system = QuerySystem(document_url=str(request.documents))
        
        logging.info(f"Processing {len(request.questions)} questions in parallel...")
        tasks = [query_system.chain.ainvoke({"input": q}) for q in request.questions]
        results = await asyncio.gather(*tasks)
        answers = [res.get("answer", "Error: Could not generate an answer.").strip() for res in results]
        
        logging.info("All questions processed successfully.")
        return SubmissionResponse(answers=answers)
        
    except Exception as e:
        logging.error(f"An error occurred during submission processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

app.include_router(router, prefix="/api/v1")