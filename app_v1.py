from ast import literal_eval
import json
from collections import defaultdict
import streamlit as st
from embeddings_generator import EmbeddingsGenerator
from data_generation import (
    generate_context_and_prompts,
    postprocess_RAG_answers,
    default_response,
)
from src.frontend.utils import (
    _get_projects_subprojects_structure,
    _custom_title,
    _add_blank_space,
    _format_interviewee,
)
from llm_multiprocessing_inference import (
    get_answers,
    get_answers_stream,
    replace_unneeded_characters,
    postprocess_structured_output,
)
import pandas as pd
import os
from streamlit_option_menu import option_menu
from src.frontend.page_overview import display_overview
from src.frontend.page_analysis import display_analysis
from src.frontend.page_chat import qa_information_retrieval
from src.frontend.page_chat_history import show_chat_history
from src.utils import _get_tags_dict

st.set_page_config(page_title="Local Docs Chat", layout="wide")

def _preprocess_interviewees(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["Interviewee Location Type"] = df["Interviewee"].apply(lambda x: ", ".join(sorted(list(set([i["location_type"] for i in x if i["location_type"]!="-"])))))
    df["Interviewee Organization Type"] = df["Interviewee"].apply(lambda x: ", ".join(sorted(list(set([i["organisation_type"] for i in x if i["organisation_type"]!="-"])))))
    df["Interviewee Gender"] = df["Interviewee"].apply(lambda x: ", ".join(sorted(list(set([i["gender"] for i in x if i["gender"]!="-"])))))
    return df

st.session_state["relevant_analysis_cols"] = [
        # "Entry Type",
        # "entry_fig_path",
        "File Name",
        "Document Publishing Date",
        "Document Source",
        "Document Title",
        "Interviewee",
        "Number of Pages",
        # "project_name",
        # "sub_project_name",
        # "docs_name",
        # "doc_id",
        # "Extracted Entries",
        # "entry_id",
        # "Embeddings",
        # "Secondary Tags",
        # "Framework Classification",
        # "Framework Probabilities",
    ]

if "chat_data" not in st.session_state:
    st.session_state["chat_data"] = pd.read_csv(
        os.path.join("data", "dataset", "entries_dataset.csv")
    ).drop_duplicates(subset=["Extracted Entries"])
    # st.session_state["chat_data"] = st.session_state["chat_data"][
    #     st.session_state["chat_data"]["Entry Type"] != "PDF Text"
    # ]
    # st.session_state["chat_data"]["Embeddings"] = st.session_state["chat_data"][
    #     "Embeddings"
    # ].apply(literal_eval)
    st.session_state["projects_subprojects_structure"] = (
        _get_projects_subprojects_structure()
    )
    if "entry_id" not in st.session_state["chat_data"].columns:
        st.session_state["chat_data"]["entry_id"] = [
            i for i in range(len(st.session_state["chat_data"]))
        ]

    for col in [
        "Interviewee",
        "Document Source",
        "Secondary Tags",
        "Framework Probabilities",
        "Embeddings",
    ]:
        st.session_state["chat_data"][col] = st.session_state["chat_data"][col].apply(
            literal_eval
        )
        
    st.session_state["chat_data"] = _preprocess_interviewees(st.session_state["chat_data"])
    st.session_state["classification_tags"] = sorted(list(st.session_state["chat_data"]["Framework Probabilities"].iloc[0].keys()))
    st.session_state["pillars"] = list(set([t.split("->")[0] for t in st.session_state["classification_tags"]]))
    st.session_state["tags_dict"] = _get_tags_dict(st.session_state["classification_tags"])
        
    st.session_state["chat_data"]["Interviewee"] = st.session_state["chat_data"]["Interviewee"].apply(_format_interviewee)
    st.session_state["chat_data"]["Document Source"] = st.session_state["chat_data"]["Document Source"].apply(lambda x: ", ".join(x))

if "additional_questions_data" not in st.session_state:
    st.session_state["additional_questions_data_path"] = os.path.join("data", "dataset", "additional_questions_data.json")
    if os.path.exists(st.session_state["additional_questions_data_path"]):
        with open(st.session_state["additional_questions_data_path"], "r") as f:
            st.session_state["additional_questions_data"] = json.load(f)
    else:
        st.session_state["additional_questions_data"] = {}
        
        
if "analyses_data" not in st.session_state:
    st.session_state["analyses_data_path"] = os.path.join("data", "dataset", "analyses_dataset.json")
    
    with open(st.session_state["analyses_data_path"], "r") as f:
        st.session_state["analyses_data"] = json.load(f)

df_relevant_columns = [
    "entry_id",
    "Entry Type",
    "Extracted Entries",
    "Document Publishing Date",
    "Document Title",
    "Document Source",
    "entry_fig_path",
]


def custom_filter_function(stream_answer):
    """
    Determine if the streaming chunk is relevant for display.

    This function analyzes partial JSON responses during streaming to determine
    if the chunk contains content from the 'answer' field that should be displayed.

    Args:
        stream_answer (str): A chunk of the streaming response

    Returns:
        bool: True if the chunk is relevant for streaming display, False otherwise
    """
    # print("stream_answer", stream_answer)
    # Check if we're in the answer field
    if '"answer":' in stream_answer:

        # If we're in the middle of the answer field but before relevancy
        if '",' in stream_answer:
            # We're still in the answer field, so it's relevant
            return False
        else:
            # print(stream_answer)
            return True

    else:
        # If we can't clearly identify that we're in the answer field,
        # consider it not relevant for streaming display
        return False


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


def display_outputs(documents_one_subproject: pd.DataFrame, project: str, subproject: str):

    _add_blank_space(1)
    tabs = option_menu(
        "",
        [
            "Overview",
            "Analysis",
            "Chat History",
            "Chat",
        ],
        icons=[
            "card-heading",
            "card-heading",
            "chat-dots",
            "chat-dots",
        ],
        styles={
            "container": {"width": "95%"},
        },
        orientation="horizontal",
        key="analysis_page",
        default_index=0,
    )

    if tabs == "Overview":
        display_overview(documents_one_subproject)

    elif tabs == "Analysis":
        display_analysis(documents_one_subproject, project, subproject)

    elif tabs == "Chat":
        qa_information_retrieval(st.session_state["chat_data"])

    elif tabs == "Chat History":
        show_chat_history()


def _format_func(x):
    return x.split("/")[-1].split(".")[0]


@st.fragment
def main_chat():
    projets_col, _, subprojects_col, _ = st.columns([0.2, 0.02, 0.2, 0.58])
    with projets_col:
        projects_selected = st.selectbox(
            "Select a project",
            sorted(list(st.session_state["projects_subprojects_structure"].keys())),
        )
    with subprojects_col:
        subprojects_selected = st.selectbox(
            "Select a subproject",
            sorted(
                st.session_state["projects_subprojects_structure"][projects_selected]
            ),
        )
    documents_one_subproject = st.session_state["chat_data"][
        (st.session_state["chat_data"]["project_name"] == projects_selected)
        & (st.session_state["chat_data"]["sub_project_name"] == subprojects_selected)
    ]

    # _add_blank_space(2)
    display_outputs(documents_one_subproject, projects_selected, subprojects_selected)


main_chat()
