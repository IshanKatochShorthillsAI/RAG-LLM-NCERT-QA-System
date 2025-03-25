# RAG-LLM-NCERT-QA-System (Ishan Katoch)

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red)](https://streamlit.io/)
[![Weaviate](https://img.shields.io/badge/Weaviate-VectorDB-brightgreen)](https://weaviate.io/)
[![Gemini API](https://img.shields.io/badge/Gemini-GenerativeAI-blueviolet)](https://ai.google.dev/)
[![Selenium](https://img.shields.io/badge/Selenium-WebAutomation-green)](https://selenium.dev/)

---

## **Project Overview**

This repository contains a complete implementation of a **Retrieval-Augmented Generation (RAG)** based Chatbot designed to answer user queries from NCERT textbooks (Grades 7 to 10). It leverages advanced NLP techniques, Gemini's Large Language Model (LLM), semantic vector retrieval with Weaviate, and a Streamlit web interface.

---

## **Key Features**

- **Automated Data Scraping:** Using Selenium to automate downloading NCERT textbooks.
- **Text Extraction and Processing:** PDFs parsed and structured into JSON format.
- **Semantic Embeddings:** Generated via Google's Gemini API for efficient context retrieval.
- **RAG Pipeline:** Semantic retrieval with Weaviate combined with Gemini LLM for accurate answers.
- **Streamlit Web UI:** User-friendly interface for interactive Q&A.
- **Comprehensive Evaluation:** Robust RAGAS evaluation, using answer_relevancy and faithfulness.

---

## **RAG Pipeline Workflow**

The pipeline consists of:

- **Data Collection:** Selenium automates NCERT PDF downloads.
- **Text Extraction:** PyPDF2 converts PDFs into structured JSON.
- **Embedding Generation:** Gemini API creates semantic embeddings.
- **Vector Storage:** Embeddings indexed efficiently in Weaviate.
- **Semantic Retrieval & Generation:** Contextual retrieval from Weaviate, responses generated using Gemini LLM.
- **Interactive Frontend:** Users interact through Streamlit apps.
  
![mermaid-diagram-2025-03-25-090901](https://github.com/user-attachments/assets/fbb4e8b7-736a-4607-9ec2-c5f8a1bcabbf)

---

## **Evaluation Framework**

Evaluated rigorously using **200+ test cases**, yielding:

| Metric              | Average Score |
|---------------------|---------------|
| Faithfulness        | 0.959         |
| Answer Relevancy    | 0.901         |
| Combined Aggregate  | 0.930         |

---

## **Dependencies**

- **Web Automation:** Selenium  
- **PDF Parsing:** PyPDF2  
- **Vector Storage:** Weaviate Cloud
- **Generative AI:** Gemini API  
- **Frontend UI:** Streamlit  
- **Evaluation Metrics:** NLTK, RAGAS, Gemini API

*(Full details in `requirements.txt`)*

By <strong>Ishan Katoch</strong>
