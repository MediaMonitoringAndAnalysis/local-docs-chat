import streamlit as st
from embeddings_generator import EmbeddingsGenerator
from data_generation import (
    generate_context_and_prompts,
    postprocess_RAG_answers,
    default_response,
)
from llm_multiprocessing_inference import get_answers

st.set_page_config(page_title="Local Docs Chat", layout="wide")

possibly_added_columns = [
    "Extraction Text",
    "Document Publishing Date",
    "Document Title",
    "Document URL",
    "Document Source",
]


embeddings_generator = EmbeddingsGenerator()


def _custom_title(
    title: str,
    margin_top: int,
    margin_bottom: int,
    font_size: int = 20,
    color: str = "black",
):
    # st.markdown(, unsafe_allow_html=True)

    st.markdown(
        f"""<div style="margin-top: {margin_top}px; margin-bottom: {margin_bottom}px; font-size: {font_size}px; color: {color}; font-weight: bold"> {title} </div>""",
        unsafe_allow_html=True,
    )


def qa_information_retrieval():

    if "qa_clicks" not in st.session_state:
        st.session_state["qa_clicks"] = 0

    # _custom_title("Ask it yourself", margin_top=5, margin_bottom=-20, font_size=40)

    _custom_title("Question", margin_top=5, margin_bottom=-20, font_size=30)
    question_input = st.text_input("Question:", label_visibility="collapsed")

    # qa_df = st.session_state["classification_data"].copy()

    if st.button("Ask Question") or question_input != "":
        st.session_state["qa_clicks"] += 1

    if st.session_state["qa_clicks"] > 0:
        with st.spinner("Processing your question... It will take 10-15 seconds."):

            question_embedding = {
                question_input: embeddings_generator([question_input])
            }

            context_df, prompts = generate_context_and_prompts(
                qa_df=qa_df,
                question_embeddings=question_embedding,
                n_kept_entries=10,
                show_progress_bar=True,
            )

            answers = get_answers(
                prompts=prompts,
                response_type="structured",
                model="gpt-4o-mini",
                default_response=default_response,
                show_progress_bar=False,
            )[0]

            # final_answer = postprocess_RAG_answers(
            #     answers=answers,
            #     context_df=context_df,
            #     text_col="Extraction Text",
            # )

            # final_answer_relevance = final_answer["relevance"]
            # final_answer_text = final_answer["answer"]

            final_answer_relevance = 0.5
            final_answer_text = "This is a test answer"

        with st.container():
            _custom_title(
                f"Key Message (Relevance: {int(100*final_answer_relevance)}%)",
                margin_top=5,
                margin_bottom=-20,
                font_size=25,
            )
            for _ in range(1):
                st.write("")
            st.markdown(final_answer_text)


qa_information_retrieval()
