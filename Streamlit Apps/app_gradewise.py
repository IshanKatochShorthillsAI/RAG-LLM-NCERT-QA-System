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
st.set_page_config(page_title="RAG Pipeline with Weaviate & Gemini", layout="wide")
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Dark Mode CSS with accents for a modern look
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
    
    /* Conversation history cards with accent borders */
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

# Define chapter names for each grade
CHAPTER_NAMES = {
    "Science Grade 7": [
        "Chapter 1 – Nutrition in Plants",
        "Chapter 2 – Nutrition in Animals",
        "Chapter 3 – Heat",
        "Chapter 4 – Acids, Bases and Salts",
        "Chapter 5 – Physical and Chemical Changes",
        "Chapter 6 – Respiration in Organisms",
        "Chapter 7 – Transportation in Animals and Plants",
        "Chapter 8 – Reproduction in Plants",
        "Chapter 9 – Motion and Time",
        "Chapter 10 – Electric Current and Its Effects",
        "Chapter 11 – Light",
        "Chapter 12 – Forests: Our Lifeline",
        "Chapter 13 – Wastewater Story",
    ],
    "Science Grade 8": [
        "Chapter 1 Crop Production and Management",
        "Chapter 2 Microorganisms: Friend and Foe",
        "Chapter 3 Combustion and Flame",
        "Chapter 4 Conservation of Plants and Animals",
        "Chapter 5 Reproduction in Animals",
        "Chapter 6 Reaching the Age of Adolescence",
        "Chapter 7 Force and Pressure",
        "Chapter 8 Friction",
        "Chapter 9 Sound",
        "Chapter 10 Chemical Effects of Electric Current",
        "Chapter 11 Some Natural Phenomena",
        "Chapter 12 Light",
    ],
    "Science Grade 9": [
        "Chapter 1 Matter in Our Surroundings",
        "Chapter 2 Is Matter Around Us Pure?",
        "Chapter 3 Atoms and Molecules",
        "Chapter 4 Structure of the Atom",
        "Chapter 5 The Fundamental Unit of Life",
        "Chapter 6 Tissues",
        "Chapter 7 Motion",
        "Chapter 8 Force and Laws of Motion",
        "Chapter 9 Gravitation",
        "Chapter 10 Work and Energy",
        "Chapter 11 Sound",
        "Chapter 12 Improvement in Food Resources",
    ],
    "Science Grade 10": [
        "Chapter 1 – Chemical reactions and equations",
        "Chapter 2 – Acids, Bases and Salt",
        "Chapter 3 – Metals and Non-metals",
        "Chapter 4 – Carbon and Its Compounds",
        "Chapter 5 – Life Processes",
        "Chapter 6 – Control and Coordination",
        "Chapter 7 – How Do Organisms Reproduce?",
        "Chapter 8 – Heredity and Evolution",
        "Chapter 9 – Light Reflection and Refraction",
        "Chapter 10 – The Human Eye and Colourful World",
        "Chapter 11 – Electricity",
        "Chapter 12 – Magnetic Effects of Electric Current",
        "Chapter 13 – Our Environment",
    ],
}

# -------------------------
# Session State Initialization for History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # List of tuples: (question, answer)


def get_history_context() -> str:
    """Build conversation history string from the last three interactions."""
    if not st.session_state.history:
        return ""
    history_lines = ["Conversation History:"]
    for q, a in st.session_state.history[-3:]:
        history_lines.append(f"Q: {q}")
        history_lines.append(f"A: {a}")
    return "\n".join(history_lines)


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


def construct_prompt(context: str, question: str) -> str:
    history_context = get_history_context()
    prompt = (
        f"{history_context}\n\n"
        "Retrieved Context:\n"
        f"{context}\n\n"
        "Based solely on the above, answer the following question in detail:\n"
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


def process_question(question: str, collection_name: str) -> tuple:
    embedding = get_embedding(question)
    if embedding is None:
        return "", ""
    client = get_weaviate_client()
    try:
        collection_obj = client.collections.get(collection_name)
    except Exception as e:
        st.error(f"Error retrieving collection {collection_name}: {e}")
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
    with st.expander("Show Retrieved Context", expanded=False):
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
# Main Application Logic
# -------------------------
def main():
    # Sidebar configuration
    st.sidebar.header("Configuration")
    grade_options = [
        "Science Grade 7",
        "Science Grade 8",
        "Science Grade 9",
        "Science Grade 10",
    ]
    selected_grade = st.sidebar.selectbox("Select Grade", grade_options)
    st.sidebar.subheader("Chapters")
    with st.sidebar.expander("View Chapters", expanded=True):
        for chapter in CHAPTER_NAMES.get(selected_grade, []):
            st.markdown(f"- {chapter}")
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Conversation History"):
        st.session_state.history = []
        st.sidebar.success("History cleared!")

    # Main title and instructions
    st.title("RAG Pipeline: Ishan (GradeWise)")
    st.markdown(
        """
        This application demonstrates a Retrieval-Augmented Generation (RAG) pipeline using Weaviate and Gemini. Put up NCERT Questions according to your grade, and get the answers.
        Enter your question below to retrieve relevant context, view the LLM prompt, and generate an in-depth answer.
        """
    )

    # Group question input and button in a form for better structure
    with st.form(key="query_form"):
        user_query = st.text_input("Enter your question:", "What is photosynthesis?")
        submit_button = st.form_submit_button(label="Run Query")

    if submit_button:
        collection_name = f"DocumentChunks_{selected_grade.replace(' ', '_')}"
        with st.spinner("Computing query embedding..."):
            query_embedding = get_embedding(user_query)
        if query_embedding is None:
            st.error("Error obtaining query embedding.")
            return
        st.success("Connected to Weaviate.")

        prompt_text, answer = process_question(user_query, collection_name)
        if not prompt_text or not answer:
            st.error("Error processing the query.")
            return

        # Collapsible container for LLM prompt (collapsed by default)
        with st.expander("View LLM Prompt", expanded=False):
            st.code(prompt_text, language="text")

        # Display the generated answer (always visible)
        st.markdown(
            "<div class='answer-card'><h3>Generated Answer</h3>" + answer + "</div>",
            unsafe_allow_html=True,
        )

        # Update conversation history (keeping last 3 interactions)
        st.session_state.history.append((user_query, answer))
        if len(st.session_state.history) > 3:
            st.session_state.history = st.session_state.history[-3:]

    # Collapsible full-width Conversation History section at the bottom
    with st.expander("View Conversation History", expanded=False):
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
