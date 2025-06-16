import streamlit as st
from typing import Dict
from chatbot import Chatbot

def render_chatbot(faq_dict: Dict[str, str]):
    """Render the chatbot component in Streamlit."""
    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize chatbot
    chatbot = Chatbot(faq_dict)
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Need help? Ask me anything!")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Use only st.chat_input (remove any st.button usage)
    if user_input := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get chatbot response
        response = chatbot.get_response(user_input)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display the new messages
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(response)
    
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun() 