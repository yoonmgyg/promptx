import streamlit as st
from openai import OpenAI
import difflib
import textstat
import os
from dotenv import load_dotenv

load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="AutoPromptX: Interactive Prompt Optimizer", layout="wide")
st.title("AutoPromptX: Interactive Prompt Optimizer")

st.markdown(
    """
    This research demo shows how LLMs can self-optimize prompts and responses.  
    **Enter any prompt below.**  
    The agent will generate an initial answer, critique and improve the prompt, then show the improvement in clarity and readability.
    """
)

def get_gpt_response(prompt, system_prompt=None, model="gpt-4o"):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def prompt_self_critique_and_rewrite(prompt, response):
    critique_instruction = f"""
You are a world-class prompt engineer. Given the original prompt and the LLM response, provide a concise critique focusing on clarity, specificity, and completeness. Then, rewrite the prompt to improve it based on your critique.

Format:
CRITIQUE: <your critique>
IMPROVED_PROMPT: <improved prompt>
    
Original Prompt:
{prompt}

LLM Response:
{response}
"""
    output = get_gpt_response(critique_instruction)
    critique = ""
    improved_prompt = ""
    if "IMPROVED_PROMPT:" in output:
        parts = output.split("IMPROVED_PROMPT:")
        critique = parts[0].replace("CRITIQUE:", "").strip()
        improved_prompt = parts[1].strip()
    else:
        improved_prompt = output.strip()
    return critique, improved_prompt

def highlight_differences(text1, text2):
    diff = difflib.ndiff(text1.split(), text2.split())
    highlighted = []
    for d in diff:
        if d.startswith("+"):
            highlighted.append(f'<span style="background-color:#d4ffd4;">{d[2:]}</span>')
        elif d.startswith("-"):
            highlighted.append(f'<span style="background-color:#ffd4d4;text-decoration:line-through;">{d[2:]}</span>')
        else:
            highlighted.append(d[2:])
    return ' '.join(highlighted)

def compute_metrics(text):
    return {
        "Readability (Flesch)": textstat.flesch_reading_ease(text),
        "Grade Level": textstat.flesch_kincaid_grade(text),
        "Sentence Count": textstat.sentence_count(text),
        "Word Count": len(text.split())
    }

# UI

user_prompt = st.text_area(
    "Enter your prompt/question:",
    value="Describe how gravity works.",
    height=100,
)

if st.button("Optimize Prompt and Analyze"):
    with st.spinner("Generating initial LLM response..."):
        initial_response = get_gpt_response(user_prompt)
    st.subheader("Step 1: Initial LLM Response")
    st.write(initial_response)

    with st.spinner("LLM analyzing and rewriting your prompt..."):
        critique, improved_prompt = prompt_self_critique_and_rewrite(user_prompt, initial_response)
    st.subheader("Step 2: LLM Self-Critique & Improved Prompt")
    st.markdown(f"**Critique:** {critique}")
    st.markdown(f"**Improved Prompt:** `{improved_prompt}`")

    with st.spinner("Generating LLM response to improved prompt..."):
        optimized_response = get_gpt_response(improved_prompt)
    st.subheader("Step 3: Optimized LLM Response")
    st.write(optimized_response)

    st.subheader("Step 4: Comparative Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Response**")
        st.markdown(initial_response)
        metrics_orig = compute_metrics(initial_response)
        st.markdown("**Metrics:**")
        st.json(metrics_orig)
    with col2:
        st.markdown("**Optimized Response**")
        st.markdown(
            highlight_differences(initial_response, optimized_response),
            unsafe_allow_html=True
        )
        metrics_opt = compute_metrics(optimized_response)
        st.markdown("**Metrics:**")
        st.json(metrics_opt)

    st.subheader("Step 5: Improvement Summary")
    st.markdown(
        f"""
        - **Readability improved by:** {metrics_opt['Readability (Flesch)'] - metrics_orig['Readability (Flesch)']:.2f}
        - **Grade Level change:** {metrics_opt['Grade Level'] - metrics_orig['Grade Level']:.2f}
        - **Word Count change:** {metrics_opt['Word Count'] - metrics_orig['Word Count']}
        """
    )

    st.info("Green highlights = new/added info; Red = removed info.")

st.markdown("---")
st.caption("AutoPromptX: LLM Agent for Automated Prompt Optimization. Research prototype for AgentX MOOC Competition.")
