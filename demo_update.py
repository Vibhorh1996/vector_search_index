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
st.markdown("# PDF Parser and Search")
st.markdown(
    "Upload one or more PDF files and click the 'Parse' button to parse them and dump the extracted data as JSON. "
    "Then, click the 'Build Index' button to create a Faiss index from the generated JSON files. "
    "Finally, enter a search query and click the 'Search' button to perform a search on the Faiss index."
)

# Set the OpenAI API key using Streamlit text input
openai_api_key = st.text_input("Enter your OpenAI API key:")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")

# Verify the OpenAI API key validity
is_valid_key = False
if openai_api_key:
    openai.api_key = openai_api_key
    try:
        response = openai.Completion.create(engine="davinci-codex", prompt="Hello", max_tokens=5)
        is_valid_key = True
    except Exception as e:
        is_valid_key = False

    if is_valid_key:
        st.success("OpenAI API key is valid.")
    else:
        st.error("Invalid OpenAI API key. Please enter a valid key to proceed.")

if is_valid_key:
    # Create checkboxes to select the GPT model
    use_gpt35 = st.checkbox("Use GPT-3.5", value=True)
    use_gpt4 = st.checkbox("Use GPT-4")

    # Determine the selected GPT model
    selected_model = "GPT-3.5" if use_gpt35 else "GPT-4"
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

    # Create a chat interface
    st.subheader("Chat with the AI")
    chat_history = []

    # Function to send a user message and get a model-generated response
    def get_model_response(message):
        if use_gpt35:
            model_name = "gpt-3.5-turbo"
        elif use_gpt4:
            model_name = "gpt-4.0-turbo"

        prompt = f"User: {message}\nAI:"
        chat_history.append(prompt)

        response = openai.Completion.create(
            engine="davinci-codex",
            prompt="\n".join(chat_history),
            temperature=0.7,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=model_name,
        )

        model_reply = response.choices[0].text.strip().split("AI:")[-1].strip()
        chat_history.append(f"AI: {model_reply}")
        return model_reply

    # Create a text input box for the user to enter messages
    user_input = st.text_input("You", "")

    if st.button("Send"):
        if user_input:
            # Get model-generated response and display it
            model_response = get_model_response(user_input)
            st.write("AI:", model_response)
        else:
            st.warning("Please enter a message.")

    # Create a search query input box and search button
    query = st.text_input("Enter a search query:")
    if st.button("Search"):
        # Search the index using the query
        indexer = FaissIndexer.load_index('./tmp.index')  # Load the index and assign it to the indexer variable

        if indexer:  # Check if index was successfully loaded
            st.markdown('**:blue[Loaded index from: tmp.index]**')
            D, I, search_results = indexer.search_index(query)  # Get distances, indices, and search results

            # Display the search results
            for i, result in enumerate(search_results):
                st.write(f"Result {i+1}: {result}")
                if i < len(search_results) - 1:  # Add a horizontal line if it's not the last result
                    st.markdown("---")
                # Display additional details about the search result if needed
        else:
            st.error("Failed to load index. Please make sure the index has been built.")
