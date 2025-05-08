import streamlit as st
import pandas as pd
from src.frontend.chat_utils import _display_chat_answer
from src.frontend.utils import _add_blank_space


@st.fragment
def show_chat_history():
    st.title("Chat History")
    
    chat_question_to_id = {
        question_data["question"]: id_
        for id_, question_data in st.session_state["additional_questions_data"].items()
    }
    
    questions = list(chat_question_to_id.keys())
    
    selected_question = st.selectbox("Select a question", questions)
    
    _add_blank_space(2)
    final_answer = st.session_state["additional_questions_data"][chat_question_to_id[selected_question]]
    
    shown_context_df = pd.DataFrame(final_answer["context_df"])
    shown_context_df.index = shown_context_df.index + 1
    st.write(
        shown_context_df.to_html(escape=False, justify="left"),
        unsafe_allow_html=True,
    )
        
    _display_chat_answer(final_answer)
    