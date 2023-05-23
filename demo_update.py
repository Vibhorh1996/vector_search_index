import streamlit as st
import pandas as pd
import os
import json
import pickle
import requests
import mimetypes
# from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as BaseFAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

# Setting page title and header
st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

# Set API Key
key = st.text_input('OpenAI API Key', '', type='password')
os.environ['OPENAPI_API_KEY'] = key
os.environ['OPENAI_API_KEY'] = key

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "DataChat", "content": "You are a helpful bot."}
    ]

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

if 'cost' not in st.session_state:
    st.session_state['cost'] = []

if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []

if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0


class FAISS(BaseFAISS):
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


class URLHandler:
    @staticmethod
    def is_valid_url(url):
        parsed_url = urlsplit(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc)

    @staticmethod
    def extract_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                absolute_url = urljoin(url, href)
                if URLHandler.is_valid_url(absolute_url) and (
                        absolute_url.startswith("http://") or absolute_url.startswith("https://")):
                    links.append(absolute_url)

        return links

    @staticmethod
    def extract_links_from_websites(websites):
        all_links = []

        for website in websites:
            links = URLHandler.extract_links(website)
            all_links.extend(links)

        return all_links


def save_uploadedfile(uploadedfile):
    with open(os.path.join("data/dataset", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return "data/dataset/" + uploadedfile.name


def train_or_load_model(documents, model_name):
    st.write(f"Training or loading FAISS index using {model_name} model...")
    loader = UnstructuredWordDocumentLoader(documents)
    embeddings = OpenAIEmbeddings()
    agent = create_pandas_dataframe_agent(ChatOpenAI, model_name, embeddings, documents)
    agent.set_custom_encoder_from_loader(loader)

    if st.button("Train"):
        agent.train(epochs=1, num_gpu=0)
        agent.save()
    else:
        agent.load()

    vector_store = FAISS()
    vector_store.load_agent(agent)
    vector_store.save("data/faiss_index.faiss")

    st.write("FAISS index has been trained and saved.")


def answer_questions(query, model_name):
    agent = create_pandas_dataframe_agent(ChatOpenAI, model_name, embeddings, documents)
    agent.load()
    vector_store = FAISS.load("data/faiss_index.faiss")
    vector_store.load_agent(agent)

    answers = vector_store.query(query)
    return answers


# Sidebar
st.sidebar.header('Data Chat Configuration')

# Model selection
st.sidebar.subheader('Select a Model')
model_name = st.sidebar.radio("Model", ("GPT-3.5", "GPT-4"))
st.session_state['model_name'] = model_name

# Clear conversation history
if st.sidebar.button('Clear Conversation'):
    st.session_state['messages'] = []
    st.session_state['total_tokens'] = 0
    st.session_state['total_cost'] = 0.0

# Main content
st.subheader('Conversation')

# Upload PDF files
uploaded_files = st.file_uploader('Upload PDF files', type=["pdf"], accept_multiple_files=True)
documents = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        document_path = save_uploadedfile(uploaded_file)
        documents.append(document_path)

# Train or load model
if documents:
    train_or_load_model(documents, model_name)

# User input and response generation
user_input = st.text_area("User Input")

if st.button("Submit"):
    # Add user message to history
    st.session_state['messages'].append({"role": "user", "content": user_input})

    # Generate response using the selected model
    if model_name == "GPT-3.5":
        api_key = os.getenv('OPENAPI_API_KEY')
        openai = OpenAI(api_key=api_key, gpt_version='gpt-3.5-turbo')
        response = openai.complete_prompt(prompt=user_input)
    else:
        response = "This feature is only available with GPT-3.5 model."

    # Add model-generated response to history
    st.session_state['messages'].append({"role": "model", "content": response})

    # Calculate and update the token count and cost
    response_tokens = len(response.split())
    total_tokens = st.session_state['total_tokens'] + response_tokens
    st.session_state['total_tokens'] = total_tokens
    cost_per_token = 0.006 if model_name == "GPT-3.5" else 0.008
    response_cost = response_tokens * cost_per_token
    total_cost = st.session_state['total_cost'] + response_cost
    st.session_state['total_cost'] = total_cost

# Display chat history
for message in st.session_state['messages']:
    if message["role"] == "user":
        st.write(f"User: {message['content']}")
    elif message["role"] == "model":
        st.write(f"Model: {message['content']}")

# Display cost and token count
st.subheader("Cost and Token Count")
st.write(f"Total Tokens: {st.session_state['total_tokens']} tokens")
st.write(f"Total Cost: ${st.session_state['total_cost']}")

