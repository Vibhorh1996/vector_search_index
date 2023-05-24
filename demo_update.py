import re
import json
import PyPDF2
import streamlit as st
from pathlib import Path
import faiss
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
        st.markdown("**:blue[ Faiss index has been built and stored at: tmp.index ]**")
    else:
        st.error("Failed to build the index. Please make sure the JSON files are parsed correctly.")

# Create search functionality
search_query = st.text_input("Enter a search query:")
if st.button("Search"):
    if search_query:
        # Search the index using the query
        indexer = FaissIndexer.load_index('./tmp.index')  # Load the index and assign it to the indexer variable

        if indexer:  # Check if index was successfully loaded
            st.markdown('**:blue[Loaded index from: tmp.index]**')

            D, I, search_results = indexer.search_index(search_query)  # Get distances, indices, and search results

            # Get the search results as a list of strings
            search_results_list = [str(result) for result in search_results]

            # Pass the search results as input to the user-provided information
            information = "\n".join(search_results_list)

            if information:
                # Initialize conversation variables
                current_topic = True  # Flag to track if the conversation is within the current topic
                continue_topic = "<Select option>"

                # Conversation loop
                while True:
                    if continue_topic == "<Select option>":
                        continue_topic = st.selectbox("Do you have any other questions on this topic?",
                                                      options=["<Select option>", "Yes", "No"],
                                                      key="continue_topic")
                    elif continue_topic == "Yes":
                        conversation_type = st.selectbox("Continue with this topic or new query?",
                                                         options=["<Select option>", "Current", "New"],
                                                         key="conversation_type")
                        if conversation_type == "Current":
                            # Display the user's original question as reference
                            st.write("Original question:", search_query)
                        elif conversation_type == "New":
                            st.write("Understood. Please provide me with a new set of information.")
                            search_query = st.text_input("Enter a new search query:")
                            if search_query:
                                st.write("Understood. You may now ask your question.")
                                current_topic = True
                                continue_topic = "<Select option>"
                            else:
                                st.warning("Please enter a new search query to proceed.")
                                current_topic = False
                        else:
                            st.warning("Invalid conversation type. Please select 'Current' or 'New'.")
                            current_topic = False
                    elif continue_topic == "No":
                        st.write("Hope you got the answer you were looking for.")
                        break
                    else:
                        st.warning("Invalid response. Please select an option.")

                    if continue_topic in ["Yes", "No"]:
                        question = st.text_input("Ask your question:", value=search_query)

                        if question:
                            # Check if the question is relevant to the given information
                            if not current_topic:
                                st.write("Please ask questions relevant to the given information only.")
                                continue

                            # Get AI-generated answer based on the provided information
                            answer = "AI-generated answer based on the provided information."
                            st.write("Answer:", answer)
                        else:
                            st.warning("Please ask a question to proceed.")
        else:
            st.error("Failed to load index. Please make sure the index has been built.")
    else:
        st.warning("Please enter a search query to proceed.")
