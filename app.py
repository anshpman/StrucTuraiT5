# app.py - Streamlit Interface for V4 QLoRA Model

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import time
import os 
import pprint
import re # Need re for parser

# --- SET PAGE CONFIG FIRST ---
# Must be the first Streamlit command
try:
    st.set_page_config(layout="wide", page_title="Wiki Structure Gen")
except st.errors.StreamlitAPIException as e:
    if "set_page_config() can only be called once per app page" in str(e):
        pass # Ignore if already set (e.g., on script rerun)
    else:
        raise # Reraise other errors

print("--- Starting Streamlit App (V4 QLoRA Model) ---")

# --- Configuration ---
BASE_MODEL_NAME = "t5-base"
# --- IMPORTANT: Path to your LOCAL adapter folder ---
# Based on your screenshot, assumes 'FINAL_ADAPTER' is a subfolder
# in the same directory as this app.py script. Adjust if needed.
LOCAL_ADAPTER_PATH = "."
TASK_PREFIX = "generate sections: "

# --- Model Loading (Cached) ---
# Use st.cache_resource to load model only once
@st.cache_resource
def load_model_and_tokenizer():
    """Loads base model, PEFT adapter, and tokenizer. Cached."""
    st.info(f"Loading Tokenizer from {LOCAL_ADAPTER_PATH}...")
    if not os.path.exists(os.path.join(LOCAL_ADAPTER_PATH, 'tokenizer_config.json')):
        st.error(f"Tokenizer files not found in {LOCAL_ADAPTER_PATH}. Please check the path and downloaded files.")
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_ADAPTER_PATH)

    st.info(f"Loading Base Model {BASE_MODEL_NAME} (attempting 8-bit)...")
    # Try loading base model in 8-bit for inference efficiency
    bnb_config_inf = BitsAndBytesConfig(load_in_8bit=True)
    try:
         # Use device_map="auto" for bitsandbytes
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config_inf,
            device_map="auto"
        )
        st.info("Base model loaded in 8-bit.")
    except Exception as e_8bit:
        st.warning(f"Could not load base model in 8-bit ({e_8bit}). Trying full precision...")
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
            # Manually determine device if not using device_map
            if torch.cuda.is_available():
                device_load = torch.device("cuda")
            else:
                device_load = torch.device("cpu")
            base_model.to(device_load)
            st.info(f"Base model loaded in full precision on {device_load}.")
        except Exception as e_full:
            st.error(f"Failed to load base model even in full precision: {e_full}")
            return None, None, None

    st.info(f"Loading PEFT adapter from {LOCAL_ADAPTER_PATH}...")
    try:
        # Load the adapter onto the base model
        peft_model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_PATH)
        # If not using device_map="auto", ensure model is on device
        if not hasattr(peft_model, 'hf_device_map'):
             if torch.cuda.is_available(): peft_model.to(torch.device("cuda"))
             else: peft_model.to(torch.device("cpu"))
        peft_model.eval() # Set to evaluation mode
        st.success(f"V4 QLoRA Model ready (on device: {peft_model.device}).")
        return peft_model, tokenizer, peft_model.device # Return the actual device model is on
    except Exception as e_peft:
        st.error(f"Error loading PEFT adapter: {e_peft}")
        return None, None, None

# --- Load model ---
model, tokenizer, device = load_model_and_tokenizer()

# --- Generation & Parsing Function ---
def generate_hierarchical_structure_peft(article_title, article_description=""):
    """Generates section structure using the PEFT model."""
    # Returns raw_string, parsed_list, error_string
    if not model or not tokenizer: # Check if model loaded successfully
        return "Model not loaded correctly.", [], "Model Load Error"
    if not article_title:
        return "Input Error: Please enter an article title.", [], "Input Error"

    input_text = f"{article_title} [SEP] {article_description}".strip()
    full_prompt = TASK_PREFIX + input_text
    print(f"Model Input Prompt: '{full_prompt}'") # Log to terminal
    start_time = time.time()

    inputs = tokenizer(full_prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']; attention_mask = inputs['attention_mask']

    raw_output = "Error during generation."
    parsed_structure = []
    error_msg = None

    try:
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=4,
                repetition_penalty=1.7, # Use the penalty that worked best
                early_stopping=True
            )
        raw_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        end_time = time.time()
        print(f"Raw Generated Output: '{raw_output}'")
        print(f"Generation time: {end_time - start_time:.2f} seconds")

        # Parsing Logic
        parse_warnings = []
        parts = raw_output.split(' | ')
        for part in parts:
            part = part.strip()
            if part.startswith('(') and part.endswith(')'):
                 content = part[1:-1]
                 if ': ' in content:
                     try: level_str, title = content.split(': ', 1); level = int(level_str); parsed_structure.append((level, title.strip()))
                     except ValueError as parse_err: parse_warnings.append(f"Could not parse part '{part}': {parse_err}")
                 else: parse_warnings.append(f"Separator ': ' not found in '{part}'")
            elif part: parse_warnings.append(f"Part '{part}' not in format '(level: title)'")
        if parse_warnings: print("Parsing Warnings:", parse_warnings) # Log warnings

    except Exception as e:
        print(f"Error during generation or parsing: {e}") # Log error
        error_msg = f"Error: {e}"
        raw_output = error_msg

    return raw_output, parsed_structure, error_msg


# --- Streamlit UI ---
st.title("Wikipedia Article Structure Generator (V4 QLoRA)")
st.markdown("Enter an article title (and optional short description) to generate a suggested section outline using the **T5-base QLoRA** model (trained on filtered data).")

# Display loading status or errors
if not model or not tokenizer:
    st.error("Model failed to load. Please check the terminal/logs for details.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        article_title_input = st.text_input("Article Title:", key="title_input", placeholder="e.g., Marie Curie")
        article_desc_input = st.text_area("Short Description (Optional):", key="desc_input", height=100, placeholder="e.g., Polish and naturalized-French physicist...")
        generate_button = st.button("Generate Structure", key="generate_button")

    with col2:
        st.subheader("Generated Structure")
        # Use session state to keep output persistent across reruns
        if 'raw_output' not in st.session_state: st.session_state.raw_output = ""
        if 'parsed_output' not in st.session_state: st.session_state.parsed_output = []
        if 'last_error' not in st.session_state: st.session_state.last_error = None

        if generate_button:
            if article_title_input:
                with st.spinner("Generating..."):
                    raw, parsed, err = generate_hierarchical_structure_peft(article_title_input, article_desc_input)
                    st.session_state.raw_output = raw
                    st.session_state.parsed_output = parsed
                    st.session_state.last_error = err
            else:
                st.warning("Please enter an Article Title.")
                st.session_state.raw_output = ""
                st.session_state.parsed_output = []
                st.session_state.last_error = None

        # Display results (or placeholder) based on session state
        if st.session_state.last_error:
            st.error(f"Generation failed: {st.session_state.last_error}")
        elif not st.session_state.raw_output and not generate_button:
             st.write("Click the button after entering a title.")
        elif not st.session_state.raw_output and generate_button and not st.session_state.last_error:
             st.write("Model generated empty output.")
        elif st.session_state.raw_output:
            st.text_area("Raw Model Output:", st.session_state.raw_output, height=100)
            st.markdown("**Parsed Hierarchical Structure:**")
            if st.session_state.parsed_output:
                display_string = ""
                for level, title in st.session_state.parsed_output:
                    # Using non-breaking spaces for indentation in markdown
                    indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * (level - 1)
                    display_string += f"{indent}- {title}<br>" # Use markdown list item + HTML line break
                st.markdown(display_string, unsafe_allow_html=True) # Allow HTML for spacing
            elif "Error" not in st.session_state.raw_output: # Don't show if raw output already shows error
                st.write("Could not parse structure from raw output.")