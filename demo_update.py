import re
import json
import PyPDF2
import streamlit as st
from pathlib import Path
import faiss
from src.parse_document import PdfParser
from src.indexer import FaissIndexer
import openai

"""
This is a Streamlit-based application that works as a frontend for building a vector search index.
"""

# Set page title and header using Streamlit Markdown
# st.set_page_config(page_title="PDF Parser and Search", page_icon=":mag:", layout="centered")
st.markdown("# PDF Parser and Search")
st.markdown(
    "Upload one or more PDF files and click the 'Parse' button to parse them and dump the extracted data as JSON. "
    "Then, click the 'Build Index' button to create a Faiss index from the generated JSON files. "
    "Finally, enter a search query and click the 'Search' button to perform a search on the Faiss index."
)

# Create checkboxes to select the GPT model
gpt_model = st.radio("Select the GPT model:", ("gpt-3.5-turbo", "GPT-4"))

# Function to get the OpenAI API key
def get_openai_api_key():
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
    return openai_api_key

# Get the OpenAI API key
openai_api_key = get_openai_api_key()

# Initialize the OpenAI API
openai.api_key = openai_api_key

# Create a file uploader for multiple files
uploaded_files = st.file_uploader(
    "Choose one or more PDF files", type="pdf", accept_multiple_files=True
)

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
    json_files = [str(Path(file.name).stem) + ".json" for file in uploaded_files]

    # Create a Faiss indexer object and build the index
    indexer = FaissIndexer(json_files)
    indexer.build_index()

    if indexer:
        indexer.save_index("./tmp.index")
        st.markdown("**:blue[ Faiss index has been built and stored at: tmp.index]**")

# Create a search query input box and search button
query = st.text_input("Enter a search query:")
if st.button("Search"):
    # Load the Faiss index
    indexer = FaissIndexer.load_index("./tmp.index")

    if indexer:
        st.markdown('**:blue[Loaded index from: tmp.index]**')
        D, I, search_results = indexer.search_index(query)

        # Prepare messages for conversation with the AI
        messages = [
            {"role": "system", "content": "You are now chatting with the AI."},
            {"role": "user", "content": query},
        ]

        # Iterate over the search results and add them to the conversation
        for i, result in enumerate(search_results):
            messages.append({"role": "assistant", "content": result})

        # Generate responses using the OpenAI language model
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=messages,
        )

        # Extract the assistant's reply from the response
        assistant_reply = response.choices[0].message.content

        # Display the AI's response
        st.markdown(f"**Q: {query}**")
        st.markdown(f"**A: {assistant_reply}**")

        # Get the user's follow-up question or response
        user_follow_up = st.text_input("Your response:")

        if user_follow_up:
            # Add the user's follow-up to the conversation
            messages.append({"role": "user", "content": user_follow_up})

            # Generate the AI's response to the follow-up
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=messages,
            )

            # Extract the assistant's reply from the response
            assistant_reply = response.choices[0].message.content

            # Display the AI's response
            st.markdown(f"**A: {assistant_reply}**")

    else:
        st.error("Failed to load index. Please make sure the index has been built.")
