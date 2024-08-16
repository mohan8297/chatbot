import streamlit as st
from model import get_answer

st.title("QA Bot")

# Add a brief description
st.write("Welcome to the QA Bot! Ask me anything, and I'll do my best to provide an answer based on the information I have.")

# User input section
user_input = st.text_input("Ask a question:")
if st.button("Submit"):
    if user_input:
        with st.spinner('Processing...'):
            try:
                answer = get_answer(user_input)  # Get the answer using the model
                st.write("Answer:", answer)
            except Exception as e:
                st.write("An error occurred:", str(e))
    else:
        st.write("Please enter a question.")
