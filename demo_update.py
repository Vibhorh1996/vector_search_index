import re
import json
import PyPDF2
import streamlit as st
from pathlib import Path
import faiss
import openai

# Set OpenAI API key
openai_api_key = st.secrets["openai_api_key"]
openai.api_key = openai_api_key

# ChatGPT prompt message
prompt_message = """
<Chat GPT, I want you to act as a Conversational AI Expert wherein a user will ask you questions, based on some information which the user will provide you. You need to answer the questions ONLY based on the information provided by the user, in at most 100 words.

Requirement from the User will be to upload the information, which can be in paragraph form, points, or any other format.

You will first prompt the user - 'Provide me with the information on which you want to ask questions on:'. Then the user will provide the information.

As the information is given to you, you will prompt the user - 'Understood. You may now ask your question.' You need to answer the question ONLY based on the information provided to you by the user.

Once you have provided the answer, in the next line within the answer only, you will prompt the user - 'Do you have any other questions on this topic?' Provide options (Y/N).

- If the user selects 'Y' or 'y', prompt the user - 'Continue with this topic or New Query?' and provide options (Current/New).
    - If the user selects 'Current' (case insensitive), prompt the user - 'Understood. You may now ask your question.' Repeat this until the user selects 'Current' (case insensitive). Make sure you ONLY answer the questions of the user based on the information which the user provided you at the first place and nothing outside of that.
    - If the user replies with 'New' (case insensitive), prompt the user - 'Understood. Please provide me with a new set of information'. At this point, the user is expected to provide new information to you. After the information is provided to you, prompt the user - 'Understood. You may now ask your question.' Again, make sure that you answer the question ONLY based on the information provided to you by the user.

- If the user asks a question outside of the provided information, prompt the user - 'Please ask questions relevant to the given information only.' and ask again - 'Do you have any other questions?' Provide options (Y/N).
- If the user replies with 'N' or 'n', prompt the user - 'Hope you got the answer you were looking for.'

Repeat the process as per the user input from the above points.
"""

# Function to generate chatbot response using OpenAI's ChatGPT
def generate_chatbot_response(message, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_message + "\nQ: " + message + "\nA:",
        temperature=0.7,
        max_tokens=50,
        n=1,
        chat_history=chat_history,
    )
    
    chat_history.append({"role": "system", "content": "Q: " + message + "\nA: " + response.choices[0].text.strip()})
    
    return response.choices[0].text.strip(), chat_history

# Remaining code...

# Set page title and header using Streamlit Markdown
st.markdown("# PDF Parser and Search")
# ...

# Create a search query input box and search button
query = st.text_input("Enter a search query:")
if st.button("Search"):
    if indexer:
        # Search the index using the query
        D, I, search_results = indexer.search_index(query)  # Get distances, indices, and search results

        chat_history = []  # Initialize chat history for the chatbot
        user_question = query
        st.markdown("Q. " + user_question)  # Display user's question

        while True:
            # Generate chatbot response
            chatbot_answer, chat_history = generate_chatbot_response(user_question, chat_history)
            st.markdown("A. " + chatbot_answer)  # Display chatbot's answer

            if re.search(r"\bnew\b", chatbot_answer, re.IGNORECASE):
                st.warning("Please provide a new set of information.")
                user_information = st.text_area("New Information:")
                st.success("Understood. You may now ask your question.")
                user_question = st.text_input("Enter your question:")
                chat_history = []  # Reset chat history for new information
            elif re.search(r"\bcurrent\b", chatbot_answer, re.IGNORECASE):
                st.success("Understood. You may now ask your question.")
                user_question = st.text_input("Enter your question:")
            else:
                continue_prompt = st.radio("Do you have any other questions on this topic?", ("Yes", "No"))
                if continue_prompt.lower() == "yes":
                    new_current_prompt = st.radio("Continue with this topic or new query?", ("Current", "New"))
                    if new_current_prompt.lower() == "current":
                        st.success("Understood. You may now ask your question.")
                        user_question = st.text_input("Enter your question:")
                    else:
                        st.warning("Please provide a new set of information.")
                        user_information = st.text_area("New Information:")
                        st.success("Understood. You may now ask your question.")
                        user_question = st.text_input("Enter your question:")
                else:
                    st.markdown("Hope you got the answer you were looking for.")
                    break

    else:
        st.error("Failed to load index. Please make sure the index has been built.")

# ...
