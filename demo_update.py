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

# Set OpenAI API key using Streamlit text input
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    openai.api_key = openai_api_key

# Set page title and header using Streamlit Markdown
st.markdown("# PDF Parser and Search")
st.markdown("Upload one or more PDF files and click the 'Parse' button to parse them and dump the extracted data as JSON. "
            "Then, click the 'Build Index' button to create a Faiss index from the generated JSON files. "
            "Finally, enter a search query and click the 'Search' button to interact with the chatbot.")

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
if st.button("Search"):
    # Search the index using the query
    indexer = FaissIndexer.load_index('./tmp.index')  # Load the index and assign it to the indexer variable

    if indexer:  # Check if index was successfully loaded
        st.markdown('**:blue[Loaded index from: tmp.index]**')
        D, I, search_results = indexer.search_index(query)  # Get distances, indices, and search results

        # Start the conversation with the chatbot
        st.write("Q:", query)

        information_provided = st.text_area("Provide me with the information on which you want to ask questions on:")

        if not information_provided:
            st.error("Please provide the necessary information.")
        else:
            continue_topic = True
            while continue_topic:
                st.write("A: [Chat GPT, I want you to act as a Conversational AI Expert wherein a user will ask you questions, based on some information which the user will provide you. You need to answer the questions ONLY based on the information provided by the user, in at most 100 words.]")

                # Chatbot responds to the user's question based on the provided information
                answer = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"Q: {query}\nA: {information_provided}\n",
                    max_tokens=100,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )

                st.write("A:", answer.choices[0].text.strip())

                another_question = st.selectbox("Do you have any other questions on this topic?", ("Yes", "No"))

                if another_question.lower() == "no":
                    st.write("A: Hope you got the answer you were looking for.")
                    continue_topic = False
                else:
                    continue_topic_input = st.selectbox("Continue with this topic or new query?", ("Current", "New"))

                    if continue_topic_input.lower() == "new":
                        information_provided = st.text_area("Understood. Please provide me with a new set of information:")
                    else:
                        st.write("A: Understood. You may now ask your question.")
                        query = st.text_input("Enter a search query:")
    else:
        st.error("Failed to load index. Please make sure the index has been built.")

# Uncomment the code below if you want to display summaries
# show_summaries = st.checkbox("Show Summaries")
# if show_summaries and uploaded_files:
#     for i, (input_filename, summary) in enumerate(summaries):
#         st.write(f"PDF: {input_filename}")
#         st.write(summary)
#         if i < len(uploaded_files) - 1:  # Add a horizontal line if it's not the last file
#             st.markdown("---")
