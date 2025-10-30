# prompt_ui.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
import streamlit as st
from logger import setup_logger
import sys

# -------------------------------
# Environment & Logger Setup
# -------------------------------
os.environ["HF_HUB_OFFLINE"] = "0"

# Initialize logger for this module
# keep=10 ensures only the 10 most recent timestamped log files are retained
logger = setup_logger("prompt_ui", keep=10)

# Load environment variables
load_dotenv(find_dotenv())
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
logger.debug(f"Hugging Face API Token loaded: {'FOUND' if api_token else 'NOT FOUND'}")

if not api_token:
    logger.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is required")

# -------------------------------
# Initialize Hugging Face Endpoint
# -------------------------------
try:
    model_name = "google/gemma-2-2b-it"
    logger.info(f"Initializing HuggingFaceEndpoint with model '{model_name}' and task 'conversational'")
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="conversational",
        huggingfacehub_api_token=api_token,
        temperature=0.7
    )
except Exception as e:
    logger.exception("Failed to initialize HuggingFaceEndpoint")
    raise

# -------------------------------
# Initialize ChatHuggingFace
# -------------------------------
try:
    logger.info("Initializing ChatHuggingFace")
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    logger.exception("Failed to initialize ChatHuggingFace")
    raise

# -------------------------------
# Execution Mode Detection
# -------------------------------
# When running under streamlit, STREAMLIT_RUN_MAIN or STREAMLIT_SERVER_PORT env var is set.
IS_STREAMLIT = bool(os.environ.get("STREAMLIT_RUN_MAIN") or os.environ.get("STREAMLIT_SERVER_PORT"))

def run_streamlit_ui():
    """Render the Streamlit interface (when run under `streamlit run`)."""
    st.header("Research Tool")
    user_input = st.text_input("Enter your prompt")
    # Log input length rather than content (avoid logging secrets)
    logger.info(f"Streamlit: current user input length: {len(user_input) if user_input is not None else 0}")
    if st.button("Summarize"):
        if not user_input:
            st.warning("Please enter a prompt first.")
            logger.info("Streamlit: user clicked Summarize with empty input.")
        else:
            try:
                logger.info("Streamlit: sending user query to model")
                response = chat_model.invoke([HumanMessage(content=user_input)])
                st.text(response.content)
                logger.info("Streamlit: Summarization successful.")
            except Exception:
                logger.exception("Streamlit: Error during model invocation")
                st.error("An error occurred while invoking the model. Check logs for details.")

def run_cli_mode():
    """Fallback CLI mode when user runs `python prompt_ui.py` directly."""
    print("\nRunning in CLI fallback mode (not Streamlit).")
    print("To run the Streamlit UI, execute: streamlit run prompt_ui.py\n")
    try:
        while True:
            user_input = input("Enter prompt (or 'quit' to exit): ").strip()
            if not user_input:
                print("Please enter a non-empty prompt.")
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Exiting.")
                break
            logger.info("CLI: sending user query to model")
            try:
                response = chat_model.invoke([HumanMessage(content=user_input)])
                print("\n=== Model Response ===")
                print(response.content)
                print("======================\n")
                logger.info("CLI: Summarization successful.")
            except Exception:
                logger.exception("CLI: Error during model invocation")
                print("An error occurred. Check the log file for details.")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting (keyboard interrupt).")
        logger.info("CLI: user terminated via keyboard interrupt.")


if IS_STREAMLIT:
    run_streamlit_ui()
else:
    logger.warning("Not running under Streamlit. Falling back to CLI mode. To use UI run: streamlit run prompt_ui.py")
    run_cli_mode()
