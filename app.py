import streamlit as st
import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
import yaml
import logging
from datetime import datetime

# YAML config
try:
    with open(r".\config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    filename=config["log_dir"] +
    f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

logger.info("Config file and logger setup completed.")

st.set_page_config(page_title="Query Excel")

# Session States
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = []

# System Prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are a helpful AI assistant who answers user questions based on the provided context, without generating any HTML code.""")

# Prompt
human_prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don"t know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""

chat_template = ChatPromptTemplate(
    messages=[system_prompt, HumanMessagePromptTemplate.from_template(human_prompt)])

# LLM
try:
    llm = ChatOllama(model=config["base_model"], base_url=config["local_url"])
except Exception as e:
    logger.error(f"Error initializing ChatOllama: {e}")
    llm = None

# Output Parser
output_parser = StrOutputParser()

# Chain
qna_chain = chat_template | llm | output_parser


# Helper Function
def llm_stream(context, question):
    try:
        if llm is None:
            yield "LLM initialization failed.  Cannot answer questions."
            return

        for event in qna_chain.stream({"context": context, "question": question}):
            yield event
    except Exception as e:
        logger.error(f"Error streaming from LLM: {e}")
        yield "An error occurred while generating the answer."


# File Uploader
with st.sidebar:
    excel_doc = st.file_uploader("Upload your Excel file (xlsx)", type=[
                                 "xlsx"], key="excel_doc")
    if excel_doc:
        try:
            pd_doc = pd.read_excel(excel_doc)
            pd_doc.to_excel(config["temporary_file_path"])
            loader = UnstructuredExcelLoader(
                config["temporary_file_path"], mode="elements")
            excel = loader.load()
            context = excel[0].metadata["text_as_html"]
            st.session_state.context = context
            st.success("Done!  Ready to ask questions.")
            logger.info(f"Document uploaded succesfuly.")

        except Exception as e:
            st.error(f"Error processing Excel file: {e}")
            logger.error(f"Error processing Excel file: {e}")
            st.warning("Please ensure your excel file is a valid .xlsx file.")


# Conversation Logic
def conversation():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(llm_stream(
                st.session_state.context, prompt))
            st.session_state.messages.append(
                {"role": "assistant", "content": message})


conversation()
