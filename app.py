import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

st.set_page_config(page_title="Multi-Agent RAG (HF only)", layout="centered")
st.title("Multi-Agent RAG Demo — Hugging Face Only")

st.caption("Using Hugging Face embeddings (all-MiniLM-L6-v2) for retrieval.")
st.write("Status: ready ✅")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Create example files if missing ---
def ensure_example_files():
    salary_path = os.path.join(DATA_DIR, "salary.txt")
    insurance_path = os.path.join(DATA_DIR, "insurance.txt")

    if not os.path.exists(salary_path):
        with open(salary_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""\
                Salary components:
                - Basic salary: fixed pay component.
                - HRA (House Rent Allowance): often 40% of basic (varies).
                - Special allowances: additional monthly allowances.
                - Deductions: provident fund, professional tax, income tax.
                How to calculate:
                - Annual gross = monthly_gross * 12
                - Annual net = annual_gross - (annual_deductions)
                """))

    if not os.path.exists(insurance_path):
        with open(insurance_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""\
                Insurance policy basics:
                - Coverage: lists what events/expenses are paid by insurer.
                - Premium: periodic payment to maintain coverage.
                - Claim process: file claim -> documents -> insurer assesses -> payout.
                - Exclusions: what is NOT covered (pre-existing, certain treatments).
                """))

ensure_example_files()

# --- Load documents ---
docs = {
    "salary": [open(os.path.join(DATA_DIR, "salary.txt"), encoding="utf-8").read()],
    "insurance": [open(os.path.join(DATA_DIR, "insurance.txt"), encoding="utf-8").read()]
}

# --- Load Hugging Face model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- Build vector store ---
def build_store():
    store = {}
    for category in docs:
        embeddings = model.encode(docs[category])
        store[category] = embeddings
    return store

vector_store = build_store()

# --- Retrieval function ---
def retrieve(query, category):
    q_emb = model.encode([query])[0]
    sims = cosine_similarity([q_emb], vector_store[category])[0]
    best_idx = np.argmax(sims)
    return docs[category][best_idx]

# --- Coordinator ---
def decide_agent(query):
    q = query.lower()
    if any(k in q for k in ["salary", "ctc", "hra", "payslip"]):
        return "salary"
    elif any(k in q for k in ["insurance", "claim", "premium", "policy"]):
        return "insurance"
    else:
        # fallback: whichever has higher similarity
        q_emb = model.encode([query])[0]
        salary_score = cosine_similarity([q_emb], vector_store["salary"])[0][0]
        ins_score = cosine_similarity([q_emb], vector_store["insurance"])[0][0]
        return "salary" if salary_score >= ins_score else "insurance"

# --- Streamlit UI ---
st.subheader("Ask a question")
user_query = st.text_area("Your question", height=120, placeholder="e.g. How do I calculate my annual salary?")

if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding best agent..."):
            agent = decide_agent(user_query)
            context = retrieve(user_query, agent)
            st.write(f"**Agent chosen:** {agent.capitalize()}")
            st.success("Answer:")
            st.write(context)
            st.caption("(Context-based answer using Hugging Face embeddings.)")
