import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from utils.rag import create_vector_store, retrieve_context
import os

# Load API Key

load_dotenv()

client = genai.Client(
api_key=os.getenv("GOOGLE_API_KEY")
)

# Streamlit UI

st.set_page_config(
page_title="AI Resume Analyzer (RAG)",
page_icon="📄"
)

st.title("📄 AI Resume Analyzer using RAG")

uploaded_file = st.file_uploader(
"Upload Resume PDF",
type=["pdf"]
)

if uploaded_file:
 pdf_reader = PdfReader(uploaded_file)

resume_text = ""

for page in pdf_reader.pages:

    text = page.extract_text()

    if text:
        resume_text += text

st.subheader("Resume Preview")

st.text_area(
    "Extracted Text",
    resume_text[:3000],
    height=250
)

if st.button("Analyze Resume"):

    try:

        with st.spinner("Creating Embeddings..."):

            vector_store = create_vector_store(
                resume_text
            )

        question = """
        Analyze this resume and provide:

        1. Key Skills
        2. Strengths
        3. Missing Skills
        4. Improvement Suggestions
        5. Resume Score out of 10
        """

        with st.spinner("Retrieving Context..."):

            context = retrieve_context(
                vector_store,
                question
            )

        prompt = f"""
        You are an expert resume reviewer.

        Use ONLY the context provided below.

        Context:
        {context}

        Question:
        {question}
        """

        with st.spinner("Generating Analysis..."):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.subheader("AI Analysis")

        st.write(response.text)

        st.download_button(
            "Download Analysis",
            response.text,
            file_name="resume_analysis.txt"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

