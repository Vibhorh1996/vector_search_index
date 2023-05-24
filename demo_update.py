import re
import json
import PyPDF2
import streamlit as st
from pathlib import Path
import faiss
import openai
from src.parse_document import PdfParser
from src.indexer import FaissIndexer

"""
This is a Streamlit-based application that works as a frontend for building a vector search index.
"""

# Set page title and header using Streamlit Markdown
# st.set_page_config(page_title="PDF Parser and Search", page_icon=":mag:", layout="centered")
st.markdown("# PDF Parser and Search")
st.markdown("Upload one or more PDF files and click the 'Parse' button to parse them and dump the extracted data as JSON. "
            "Then, click the 'Build Index' button to create a Faiss index from the generated JSON files. "
            "Finally, enter a search query and click the 'Search' button to perform a search on the Faiss index.")

# Set the OpenAI API key using Streamlit text input
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")

# # Create checkboxes to select the GPT model
# use_gpt35 = st.checkbox("Use GPT-3.5", value=True)
# use_gpt4 = st.checkbox("Use GPT-4")

# Create radio buttons to select the GPT model
selected_model = st.radio("Select GPT model:", options=["GPT-3.5", "GPT-4"])
st.write(f"Selected GPT model: {selected_model}")

# Create a file uploader for multiple files
uploaded_files = st.file_uploader("Choose one or more PDF files", type="pdf", accept_multiple_files=True)

# Create a button to start parsing
if st.button("Parse"):
    if uploaded_files:
        summaries = []  # Store summaries for each PDF file
        for uploaded_file in uploaded_files:
            # Create a PdfParser object for each uploaded file
            pdf_parser = PdfParser(uploaded_file)

            # Parse the PDF file
            pdf_parser.parse_pdf()

            # Write the extracted data to a JSON file with the same name as the PDF file
            input_filename = Path(uploaded_file.name)
            output_filename = input_filename.stem + ".json"
            pdf_parser.write_json(output_filename)

            # Extract a short brief from the PDF
            # Modify this section to extract the summary/brief from the parsed PDF
            summary = "Summary: [Summary of the PDF in at most 50 words]"
            summaries.append((input_filename, summary))

            # Display a success message
            st.success(f"Extracted data from {input_filename} has been written to {output_filename}.")
    else:
        # Display an error message if no file was uploaded
        st.error("Please upload one or more PDF files.")

# Create a button to build the Faiss index
if st.button("Build Index"):
    # Define the list of JSON files to index
    json_files = [str(Path(file.name).stem) + '.json' for file in uploaded_files]

    # Create a Faiss indexer object and build the index
    indexer = FaissIndexer(json_files)
    indexer.build_index()

    if indexer:
        indexer.save_index('./tmp.index')
        st.markdown("**:blue[ Faiss index has been built and stored at: tmp.index]**")

# Create a search query input box and search button
query = st.text_input("Enter a search query:")

# Check if the user has entered a query and the Faiss index exists
if query and Path('./tmp.index').is_file():
    # Load the Faiss index
    indexer = FaissIndexer.load_index('./tmp.index')

    if indexer:  # Check if index was successfully loaded
        st.markdown('**:blue[Loaded index from: tmp.index]**')
        D, I, search_results = indexer.search_index(query)  # Get distances, indices, and search results

        # Store the search results to be used in the conversation
        conversation_info = {'query': query, 'search_results': search_results}

        # Display the search results
        for i, result in enumerate(search_results):
            st.write(f"Result {i+1}: {result}")
            if i < len(search_results) - 1:  # Add a horizontal line if it's not the last result
                st.markdown("---")
            # Display additional details about the search result if needed

        # Prompt the user for information
        st.text("Provide me with the information on which you want to ask questions on:")
        information = st.text_area("")

        # Check if the user has provided information
        if information:
            st.text("Understood. You may now ask your question.")

            # Create a conversation list to store the user and AI responses
            conversation = []

            # Start the conversation loop
            while True:
                user_question = st.text_input("User:")
                if user_question:
                    # Add the user question to the conversation
                    conversation.append(user_question)

                    # Check if the user question is related to the provided information
                    if user_question.strip().endswith("?") and information.lower() in user_question.lower():
                        # Get the AI response based on the user question
                        ai_response = get_ai_response(user_question, conversation_info, selected_model, openai_api_key)

                        # Add the AI response to the conversation
                        conversation.append(ai_response)

                        # Display the AI response
                        st.text("AI:" + ai_response)

                        # Check if the conversation should continue
                        continue_conversation = st.selectbox("Do you have any other questions on this topic?", options=["Yes", "No"])

                        if continue_conversation.lower() == "no":
                            st.text("Hope you got the answer which you were looking for.")
                            break

                        # Check if the user wants to continue with the current topic or start a new query
                        continue_topic = st.selectbox("Continue with this topic or new query?", options=["Current", "New"])

                        if continue_topic.lower() == "new":
                            st.text("Understood. Please provide me with a new set of information.")
                            information = st.text_area("")
                            conversation_info['search_results'] = []  # Clear the search results for the new information
                            conversation = []  # Clear the conversation for the new information
                            st.text("Understood. You may now ask your question.")
                        else:
                            st.text("Understood. You may now ask your question.")
                    else:
                        st.text("Please ask questions relevant to the given information only.")
                else:
                    st.text("Please ask a question.")

    else:
        st.error("Failed to load index. Please make sure the index has been built.")
else:
    st.text("Please enter a search query.")

# Function to get AI response
def get_ai_response(question, conversation_info, selected_model, openai_api_key):
    context = f"Question: {question}\nInformation: {conversation_info['search_results']}"
    prompt = f"<Chat GPT, i want you to act as a Conversational AI Expert wherein a user will ask you questions, based on some information which the user will provide you. You need to answer the questions ONLY based on the information provided by the user, in at most 100 words.\n\nUser Question: {question}\n\nAnswer:"

    # Send the prompt to OpenAI API to get the AI response
    ai_response = generate_ai_response(prompt, context, selected_model, openai_api_key)

    # Extract the answer from the AI response
    answer = extract_answer(ai_response)

    return answer

# Function to generate AI response using OpenAI API
def generate_ai_response(prompt, context, selected_model, openai_api_key):
    # Set the GPT model name
    model_name = "gpt-3.5-turbo" if selected_model == "GPT-3.5" else "gpt-4.0-turbo"

    # Set the OpenAI API endpoint
    api_endpoint = "https://api.openai.com/v1/engines/davinci-codex/completions" if model_name == "gpt-4.0-turbo" else "https://api.openai.com/v1/engines/davinci/completions"

    # Set the OpenAI API headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Set the OpenAI API data
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 1.0,
        "n": 1,
        "stop": None,
        "context": context,
    }

    # Send the request to OpenAI API and get the response
    response = requests.post(api_endpoint, headers=headers, json=data)
    response_json = response.json()

    # Extract the AI response from the API response
    ai_response = response_json['choices'][0]['text']

    return ai_response

# Function to extract the answer from AI response
def extract_answer(ai_response):
    # Remove the prompt and leading/trailing whitespaces from the AI response
    answer = ai_response.split("Answer:")[1].strip()

    return answer
