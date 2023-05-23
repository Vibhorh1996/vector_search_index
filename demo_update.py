import json
import PyPDF2
import streamlit as st
from pathlib import Path
import openai
import time

"""
This is a Streamlit-based application that allows users to ask questions to the AI based on uploaded PDF files.
"""

# Set page title and header using Streamlit Markdown
st.markdown("# PDF Chatbot")
st.markdown("Upload one or more PDF files and ask questions to get responses from the AI.")

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

# Create a button to start the chatbot
if st.button("Start Chatbot"):
    if uploaded_files:
        # Process each uploaded PDF file
        for uploaded_file in uploaded_files:
            st.markdown(f"## PDF: {uploaded_file.name}")
            
            # Read the PDF file
            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
            num_pages = pdf_reader.numPages

            # Extract text from each page
            pdf_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.getPage(page_num)
                pdf_text += page.extractText()

            # Prepare messages for conversation with the AI
            messages = [
                {"role": "system", "content": "You are now chatting with the AI."},
            ]

            # Ask questions and get responses from the AI
            while True:
                user_query = st.text_input(
                    "User:",
                    key=f"query_{uploaded_file.name}_{time.time()}"
                )
                if user_query:
                    # Add user query to the conversation
                    messages.append({"role": "user", "content": user_query})

                    # Generate AI response using OpenAI language model
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=messages,
                        max_tokens=50,
                        n=1,
                        stop=None,
                        temperature=0.7,
                    )

                    # Extract the AI's reply from the response
                    ai_reply = response.choices[0].text.strip()

                    # Add AI reply to the conversation
                    messages.append({"role": "assistant", "content": ai_reply})

                    # Display AI's reply
                    st.text_area("AI:", value=ai_reply, key=f"reply_{uploaded_file.name}_{time.time()}")

                if st.button(f"End Chat_{uploaded_file.name}_{time.time()}"):
                    break

    else:
        # Display an error message if no file was uploaded
        st.error("Please upload one or more PDF files.")
