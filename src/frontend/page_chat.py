import streamlit as st
import pandas as pd
import json
from embeddings_generator import EmbeddingsGenerator
from data_generation import (
    generate_context_and_prompts,
    postprocess_RAG_answers,
    default_response,
)
from src.frontend.utils import _custom_title, _compress_string
from llm_multiprocessing_inference import (
    get_answers_stream,
)
from src.frontend.chat_utils import _display_chat_answer
embeddings_generator = EmbeddingsGenerator()


df_relevant_columns = [
    "entry_id",
    "Entry Type",
    "Extracted Entries",
    "Document Publishing Date",
    "Document Title",
    "Document Source",
    "entry_fig_path",
]


def _generate_answers(prompts, context_df):
    answers_stream, answers = get_answers_stream(
        prompts=prompts,
        response_type="structured",
        api_pipeline="Ollama",
        model="deepseek-r1:14b-qwen-distill-q4_K_M",
        default_response=default_response,
        custom_filter_function=None,
        # show_progress_bar=False
    )
    st.write_stream(answers_stream)

    # answers = get_answers(
    #     prompts=prompts,
    #     response_type="structured",
    #     # api_pipeline="Ollama",
    #     # model="llama3.1:8b-text-q5_K_M",
    #     default_response=default_response,
    #     # show_progress_bar=False,
    #     api_pipeline="OpenAI",
    #     model="gpt-4o-mini",
    #     api_key=os.getenv("openai_api_key"),
    # )

    # st.markdown(answers)

    # if answers["answer"] != "-":
    final_answer = postprocess_RAG_answers(
        answers=answers,
        context_df=context_df,
        df_relevant_columns=df_relevant_columns,
    )[0]
    # else:
    #     final_answer = {
    #         "final_answer": "-",
    #         "final_relevance": 0.0,
    #         "final_context": [],
    #     }

    # st.markdown(final_answer)

    return final_answer


def _generate_chat_answer(question_input, qa_df):
    with st.spinner("Processing your question (this may take up to 30 seconds)..."):

        question_embedding = {
            question_input: embeddings_generator([question_input])
        }

        prompts, context_df = generate_context_and_prompts(
            qa_df=qa_df,
            question_embeddings=question_embedding,
            n_initial_kept_entries=200,
            n_kept_entries=5,
            zero_shot_reranking_pipeline={
                "first_pass_model": "MoritzLaurer/bge-m3-zeroshot-v2.0",
                "do_second_pass": False,
                "first_pass_threshold": None,
                "batch_size": 8,
                "device": None,
            },
            text_col="Extracted Entries",
            # show_progress_bar=True,
        )

    st.markdown("## Context used for the answer")
    shown_context_df = (
        context_df[context_df["relevance"] > 0.6]
        .reset_index(drop=True)
        .copy()[
            st.session_state["relevant_analysis_cols"]
            + ["Extracted Entries", "relevance"]
        ]
    )
    shown_context_df.index = shown_context_df.index + 1
    shown_context_df["relevance"] = shown_context_df["relevance"].apply(
        lambda x: f"{100*x:.0f}%"
    )
    
    st.write(
        shown_context_df.rename(columns={"relevance": "Relevance Score"}).to_html(escape=False, justify="left"),
        unsafe_allow_html=True,
    )

    _custom_title("Reasoning...", margin_top=5, margin_bottom=-20, font_size=25)
    final_answer = _generate_answers(prompts, context_df)

    return shown_context_df, final_answer


@st.fragment
def qa_information_retrieval(qa_df):

    # st.dataframe(qa_df)

    if "qa_clicks" not in st.session_state:
        st.session_state["qa_clicks"] = 0

    # _custom_title("Ask it yourself", margin_top=5, margin_bottom=-20, font_size=40)

    _custom_title("Question", margin_top=5, margin_bottom=-20, font_size=30)
    question_input = st.text_input("Question:", label_visibility="collapsed")

    breakdown_metadata = [
        "No Breakdown",
        "Interviewee Location Type",
        "Interviewee Organization Type",
        "Interviewee Gender",
    ]

    qa_df = st.session_state["chat_data"].copy()

    if st.button("Ask Question") or question_input != "":
        st.session_state["qa_clicks"] += 1

    if st.session_state["qa_clicks"] > 0:
        qa_chat_hash = _compress_string(question_input.lower().strip())
        
        if qa_chat_hash not in st.session_state["additional_questions_data"]:
            shown_context_df, final_answer = _generate_chat_answer(question_input, qa_df)
            final_answer["question"] = question_input
            final_answer["context_df"] = shown_context_df.to_dict(orient="records")
            st.session_state["additional_questions_data"][qa_chat_hash] = final_answer
            with open(st.session_state["additional_questions_data_path"], "w") as f:
                json.dump(st.session_state["additional_questions_data"], f)
        else:
            final_answer = st.session_state["additional_questions_data"][qa_chat_hash]
            shown_context_df = pd.DataFrame(final_answer["context_df"])
            shown_context_df.index = shown_context_df.index + 1
            st.write(
                shown_context_df.rename(columns={"relevance": "Relevance Score"}).to_html(escape=False, justify="left"),
                unsafe_allow_html=True,
            )
            
        _display_chat_answer(final_answer)