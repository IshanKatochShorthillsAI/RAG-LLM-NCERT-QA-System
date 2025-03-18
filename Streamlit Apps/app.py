import os
import logging
import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types

# -------------------------
# Configuration and Logging
# -------------------------
st.set_page_config(
    page_title="RAG Pipeline with Weaviate & Gemini (Unified Collection)", layout="wide"
)
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Use a unified collection for all documents
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "DocumentChunks_All")

# -------------------------
# Initialize Session State for History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # List of tuples: (question, answer)

# -------------------------
# Dark Mode CSS with Accents and Collapsible Sections
# -------------------------
st.markdown(
    """
    <style>
    /* Global dark styling with accent colors */
    body {
        background: linear-gradient(135deg, #121212, #1e1e1e);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #e0e0e0;
    }
    
    /* Hide the hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .css-18e3th9 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #f0f0f0;
    }
    
    /* Prompt container with accent border */
    .prompt-container {
        background-color: #2a2a2a;
        border: 2px solid #8a2be2;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Generated answer card with accent shadow */
    .answer-card {
        background-color: #1f1f1f;
        border: 2px solid #8a2be2;
        color: #e0e0e0;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0px 6px 8px rgba(138, 43, 226, 0.4);
    }
    
    /* Conversation history card styling */
    .history-card {
        background: linear-gradient(135deg, #2a2a2a, #1e1e1e);
        border: 2px solid #8a2be2;
        color: #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 6px rgba(138, 43, 226, 0.3);
        transition: transform 0.2s;
    }
    .history-card:hover {
        transform: scale(1.02);
    }
    
    /* Retrieved context sections with accent border */
    .context-section {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        color: #e0e0e0;
    }
    
    /* Custom button styling with rounded corners and accent */
    .stButton>button {
        background-color: #8a2be2;
        color: #ffffff;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 12px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #7a1ac1;
        transform: translateY(-2px);
    }
    
    /* Sidebar dark mode styling */
    .sidebar .css-1d391kg {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 12px;
        color: #e0e0e0;
    }
    .sidebar .css-1d391kg h2,
    .sidebar .css-1d391kg h3 {
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Weaviate Client Initialization
# -------------------------
@st.cache_resource(show_spinner=True)
def get_weaviate_client() -> weaviate.Client:
    WEAVIATE_URL = os.getenv(
        "WEAVIATE_URL",
        "https://oxhv5pqtk6xl62xb6gq.c0.asia-southeast1.gcp.weaviate.cloud",
    )
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
    )
    return client


# -------------------------
# Gemini Embedding Function
# -------------------------
@st.cache_data(show_spinner=False)
def get_embedding(input_text: str) -> list:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key not provided. Please set GEMINI_API_KEY.")
        return None
    genai.configure(api_key=gemini_api_key)
    model_name = "models/embedding-001"
    response = genai.embed_content(
        model=model_name,
        content=input_text,
        task_type="retrieval_document",
        title="Custom query",
    )
    if "embedding" not in response:
        st.error("Error obtaining embedding from Gemini.")
        return None
    return response["embedding"]


# -------------------------
# Helper Functions
# -------------------------
def clean_text(text: str) -> str:
    return " ".join(text.split())


def get_history_context() -> str:
    """Construct a context string from the last three interactions."""
    if not st.session_state.history:
        return ""
    history_lines = ["Conversation History:"]
    for q, a in st.session_state.history[-3:]:
        history_lines.append(f"Q: {q}")
        history_lines.append(f"A: {a}")
    return "\n".join(history_lines)


def construct_prompt(context: str, question: str) -> str:
    history_context = get_history_context()
    prompt = (
        f"{history_context}\n\n"
        "Retrieved Context:\n"
        f"{context}\n\n"
        "Based solely on the above information, answer the following question in detail:\n"
        f"Question: {question}\nAnswer:"
    )
    return prompt


def get_llm_answer(prompt: str) -> str:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key not provided. Please set GEMINI_API_KEY.")
        return ""
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        config = types.GenerationConfig(
            temperature=0.45,
            top_p=0.9,
            top_k=40,
            max_output_tokens=300,
        )
        response = model.generate_content(prompt, generation_config=config)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        logging.exception("Error generating answer from Gemini.")
        return ""


def process_question(question: str) -> tuple:
    embedding = get_embedding(question)
    if embedding is None:
        return "", ""
    client = get_weaviate_client()
    try:
        collection_obj = client.collections.get(COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error retrieving collection {COLLECTION_NAME}: {e}")
        return "", ""
    try:
        response = collection_obj.query.near_vector(
            embedding,
            limit=3,
            return_metadata=MetadataQuery(distance=True),
        )
    except Exception as e:
        st.error(f"Query failed: {e}")
        return "", ""
    retrieved_texts = []
    with st.expander("View Retrieved Context", expanded=False):
        for obj in response.objects:
            props = obj.properties
            st.markdown(
                f"<div class='context-section'><strong>Text:</strong> {clean_text(props.get('text', ''))[:300]}...</div>",
                unsafe_allow_html=True,
            )
            st.write(f"**Grade:** {props.get('grade', 'N/A')}")
            st.write(f"**Source:** {props.get('source', 'N/A')}")
            st.write(f"**Chunk:** {props.get('chunk_index', 'N/A')}")
            st.write(f"**Distance:** {obj.metadata.distance:.3f}")
            st.markdown("---")
            retrieved_texts.append(clean_text(props.get("text", "")))
    context = "\n\n".join(retrieved_texts)
    prompt = construct_prompt(context, question)
    answer = get_llm_answer(prompt)
    return prompt, answer


# -------------------------
# Main Application Logic (Unified Collection)
# -------------------------
def main():
    # Sidebar: Configuration & History Controls
    st.sidebar.header("Configuration")
    st.sidebar.markdown("**Unified Collection:**")
    st.sidebar.code(COLLECTION_NAME)
    if st.sidebar.button("Clear Conversation History"):
        st.session_state.history = []
        st.sidebar.success("History cleared!")

    # Main Title and Intro
    st.title("RAG Pipeline : Ishan (Unified Collection)")
    st.write("Using unified collection:", COLLECTION_NAME)

    # Group question input and button in a form
    with st.form(key="query_form"):
        user_query = st.text_input("Enter your question:", "What is photosynthesis?")
        submit_button = st.form_submit_button(label="Run Query")

    if submit_button:
        with st.spinner("Computing query embedding..."):
            query_embedding = get_embedding(user_query)
        if query_embedding is None:
            st.error("Error obtaining query embedding.")
            return
        st.success("Connected to Weaviate.")

        prompt_text, answer = process_question(user_query)
        if not prompt_text or not answer:
            st.error("Error processing the query.")
            return

        # Collapsible container for the LLM prompt (collapsed by default)
        with st.expander("View LLM Prompt", expanded=False):
            st.code(prompt_text, language="text")

        # Always-visible Generated Answer
        st.markdown(
            "<div class='answer-card'><h3>Generated Answer</h3>" + answer + "</div>",
            unsafe_allow_html=True,
        )

        # Update conversation history (keep last 3 interactions)
        st.session_state.history.append((user_query, answer))
        if len(st.session_state.history) > 3:
            st.session_state.history = st.session_state.history[-3:]

    # Collapsible full-width Conversation History at the bottom
    with st.expander("View Conversation History (Last 3)", expanded=False):
        if st.session_state.history:
            for i, (q, a) in enumerate(st.session_state.history, start=1):
                st.markdown(
                    f"<div class='history-card'><strong>{i}. Q:</strong> {q}<br/><strong>A:</strong> {a}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No conversation history available.")


if __name__ == "__main__":
    main()
