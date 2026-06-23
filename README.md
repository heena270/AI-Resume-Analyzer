# AI Resume Analyzer using RAG

## Overview

An AI-powered Resume Analyzer that uses Retrieval-Augmented Generation (RAG) to evaluate resumes and provide intelligent feedback.

## Features

* PDF Resume Upload
* Text Extraction using PyPDF
* Chunking of Resume Content
* Embedding Generation using SentenceTransformers
* Semantic Retrieval
* Gemini AI Analysis
* Resume Scoring and Recommendations

## Tech Stack

* Python
* Streamlit
* PyPDF
* SentenceTransformers
* NumPy
* Gemini API

## RAG Pipeline

PDF Resume
→ Text Extraction
→ Chunking
→ Embeddings
→ Similarity Search
→ Context Retrieval
→ Gemini AI
→ Resume Analysis

## Run

pip install -r requirements.txt

streamlit run app.py
