# Intelligent Query-Retrieval System for HackRx 6.0

This repository contains the complete solution for the HackRx 6.0 hackathon challenge: an LLM-Powered Intelligent Query–Retrieval System. The system is designed to process large policy documents, understand natural language questions, and provide accurate, concise, and fast answers based on the document's content.

This solution is engineered to excel across all evaluation criteria: **Accuracy, Latency, Token Efficiency, Reusability, and Explainability.**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-blue)](https://fastapi.tiangolo.com/)
[![LLM](https://img.shields.io/badge/LLM-Groq%20Llama3--8B-green)](https://groq.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-HuggingFace-yellow)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![VectorDB](https://img.shields.io/badge/VectorDB-FAISS%20(CPU)-orange)](https://github.com/facebookresearch/faiss)

## Table of Contents

- [Features](#features)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running Locally](#running-locally)
  - [Deployment for Submission](#deployment-for-submission)
- [API Documentation](#api-documentation)
  - [Endpoint](#endpoint)
  - [Authentication](#authentication)
  - [Sample Request](#sample-request)
- [Design Choices & Optimizations](#design-choices--optimizations)

## Features

-   **Multi-Format Document Processing:** Handles PDF and DOCX files from a URL.
-   **High-Speed, Low-Latency:** Optimized for sub-10-second responses for multiple queries using parallel processing (`asyncio`) and a high-speed LLM (`Groq Llama3-8B`).
-   **High-Accuracy RAG Pipeline:** Implements a Retrieval-Augmented Generation (RAG) pipeline using state-of-the-art local embeddings and a powerful LLM.
-   **Cost-Effective & Efficient:** Utilizes local embeddings to eliminate API costs for vectorization and a free-tier-friendly LLM to manage operational costs and avoid rate-limiting.
-   **Explainable & Concise Answers:** Advanced prompt engineering ensures answers are direct, synthesized, factually precise, and free of conversational filler.
-   **Scalable & Reusable:** The core logic is encapsulated in a modular `QuerySystem` class, making the code clean and easy to extend.

## System Architecture & Workflow

The system follows a modern RAG pipeline architecture:

1.  **Input:** The API receives a POST request with a document URL and a list of questions.
2.  **Document Loading:** The document is downloaded from the URL and loaded into memory.
3.  **Text Splitting:** The document is split into smaller, overlapping chunks for effective embedding.
4.  **Embedding Generation:** Each chunk is converted into a numerical vector using a local `sentence-transformers` model. This is a CPU-bound process.
5.  **Vector Store Indexing:** The embeddings are stored in an in-memory `FAISS` index for ultra-fast similarity search.
6.  **Parallel Retrieval & Generation:**
    -   For each question, a `FAISS` search retrieves the most relevant document chunks.
    -   All questions are processed **in parallel** using `asyncio.gather`.
    -   The retrieved context and the question are sent to the `Groq Llama3-8B` model using a highly optimized prompt.
7.  **JSON Output:** The model's answers are collected and returned in a structured JSON response.

## Tech Stack

| Component               | Technology                                                                          | Rationale                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Backend Framework**   | [FastAPI](https://fastapi.tiangolo.com/)                                            | High-performance, async support, automatic documentation.                        |
| **Language Model (LLM)**| [Groq Llama3-8B](https://groq.com/)                                                   | Extreme speed (tokens/sec) to meet latency goals and stay within free-tier limits. |
| **Embeddings**          | [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | High-quality, fast, local (no API cost), and CPU-friendly.                         |
| **Vector Database**     | [FAISS (CPU)](https://github.com/facebookresearch/faiss)                            | In-memory, incredibly fast for on-the-fly indexing, no external DB required.     |
| **Async Processing**    | [asyncio](https://docs.python.org/3/library/asyncio.html)                           | Enables parallel processing of all questions, drastically reducing total response time. |

## Getting Started

### Prerequisites

-   Python 3.9 or higher
-   A Groq API Key (get one at [console.groq.com](https://console.groq.com/keys))
-   The HackRx 6.0 Team Token
-   A GitHub account

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The first run of `pip install` will download `torch` and `sentence-transformers` models, which may take a few moments).*

### Configuration

Create a `.env` file in the root of the project and add your secret keys. **This is a mandatory step.**

```env
# .env

# IMPORTANT: Replace with your personal Groq API key
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# The team token provided in the hackathon problem statement
HACKRX_TEAM_TOKEN="916f89e4fb9665de950857a622a9dfa58b8311919f6f2bf0392194c2d9a711db"
