import sys
import os

# Set Python path to access src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
st.set_page_config(page_title="smllr Prompt Optimizer", layout="wide")

# Now import your modules
from src.glove_loader import load_glove_model
from src.pif_generator import generate_pif
from src.token_estimator import count_tokens, estimate_cost, sustainability_estimate
from src.prompt_cleaner import build_optimized_prompt
# from st_theme import get_theme
# theme = get_theme()
# src/ui/app.py

# Load GloVe once
@st.cache_resource
def load_embeddings():
    return load_glove_model()

glove = load_embeddings()

# col1, col2, col3 = st.columns([1, 4, 1])

# with col2:
#     st.markdown("""
#         <div style='display: flex; align-items: center; justify-content: center; gap: 12px;'>
#             <img src='assets/smllr-logo-light.png' width='60' />
#             <h1 style='margin: 0; font-weight: 700;'>smllr – Prompt Compiler & Token Optimizer</h1>
#         </div>
#     """, unsafe_allow_html=True)
#     if theme == "Dark":
#         st.image("assets/smllr-logo-light.png", width=200)
#     else:
#         st.image("assets/smllr-logo-dark.png", width=200)

#     st.markdown("<h1 style='text-align: center; margin-top: -20px;'>smllr – Prompt Compiler & Token Optimizer</h1>", unsafe_allow_html=True)

st.title("smllr – Prompt Compiler & Token Optimizer")

prompt_input = st.text_area("✏️ Enter your original LLM prompt:", height=200)

if st.button("⚡ Optimize Prompt"):
    if not prompt_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        tokens_before = count_tokens(prompt_input)
        cost_before = estimate_cost(tokens_before)
        energy_before = sustainability_estimate(tokens_before)

        pif = generate_pif(prompt_input, glove, intent="auto", output="text")
        pif_str = str(pif)

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                with st.expander(" ", expanded=True):
                    st.subheader("Original Prompt")
                    st.code(prompt_input, language='text')
                    st.metric("Tokens", tokens_before)
                    st.metric("💲 Cost ($)", f"{cost_before}")
                    st.metric("🌱 Energy Use", energy_before)

            with col2:
                with st.expander(" ", expanded=True):
                    optimized_prompt = build_optimized_prompt(pif)
                    optimized_tokens = count_tokens(optimized_prompt)
                    st.subheader("Optimized Prompt")
                    st.code(optimized_prompt)
                    st.metric("Tokens After", optimized_tokens)
                    cost_after = estimate_cost(optimized_tokens)
                    st.metric("💲 Cost After", f"{cost_after}")
                    energy_after = sustainability_estimate(optimized_tokens)
                    st.metric("🌱 Energy After", energy_after)

        st.success(f"🎯 Tokens saved: {tokens_before - optimized_tokens} | Cost saved: ${round(cost_before - cost_after, 5)}")
