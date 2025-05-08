from ast import literal_eval
import streamlit as st
from embeddings_generator import EmbeddingsGenerator
from data_generation import (
    generate_context_and_prompts,
    postprocess_RAG_answers,
    default_response,
)
from src.frontend.utils import _get_projects_subprojects_structure, _custom_title
from llm_multiprocessing_inference import (
    get_answers,
    get_answers_stream,
    replace_unneeded_characters,
    postprocess_structured_output,
)
import pandas as pd
import os

st.set_page_config(page_title="Local Docs Chat", layout="wide")


if "chat_data" not in st.session_state:
    st.session_state["chat_data"] = pd.read_csv(
        os.path.join("data", "dataset", "entries_dataset.csv")
    )
    # st.session_state["chat_data"] = st.session_state["chat_data"][
    #     st.session_state["chat_data"]["Entry Type"] != "PDF Text"
    # ]
    st.session_state["chat_data"]["Embeddings"] = st.session_state["chat_data"][
        "Embeddings"
    ].apply(literal_eval)
    st.session_state["projects_subprojects_structure"] = (
        _get_projects_subprojects_structure()
    )
    if "entry_id" not in st.session_state["chat_data"].columns:
        st.session_state["chat_data"]["entry_id"] = [
            i for i in range(len(st.session_state["chat_data"]))
        ]

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


@st.fragment
def qa_information_retrieval(qa_df):

    # st.dataframe(qa_df)

    if "qa_clicks" not in st.session_state:
        st.session_state["qa_clicks"] = 0

    # _custom_title("Ask it yourself", margin_top=5, margin_bottom=-20, font_size=40)

    _custom_title("Question", margin_top=5, margin_bottom=-20, font_size=30)
    question_input = st.text_input("Question:", label_visibility="collapsed")

    # qa_df = st.session_state["classification_data"].copy()

    if st.button("Ask Question") or question_input != "":
        st.session_state["qa_clicks"] += 1

    if st.session_state["qa_clicks"] > 0:
        with st.spinner("Processing your question..."):

            question_embedding = {
                question_input: embeddings_generator([question_input])
            }

            prompts, context_df = generate_context_and_prompts(
                qa_df=qa_df,
                question_embeddings=question_embedding,
                n_kept_entries=5,
                text_col="Extracted Entries",
                # show_progress_bar=True,
            )

        _custom_title("Reasoning...", margin_top=5, margin_bottom=-20, font_size=25)
        final_answer = _generate_answers(prompts, context_df)

        with st.container():

            for _ in range(1):
                st.write("")

            answer_col, context_col = st.columns([0.4, 0.6])
            with answer_col:
                final_answer_relevance = final_answer["final_relevance"]
                _custom_title(
                    f"Key Message (Confidence: {int(100*final_answer_relevance)}%)",
                    margin_top=5,
                    margin_bottom=-20,
                    font_size=25,
                )

                final_answer_text = final_answer["final_answer"]
                st.markdown(final_answer_text)

                # st.markdown(f"**Confidence: {int(100*final_answer_relevance)}%**")
                # st.markdown(final_answer_text)

            with context_col:
                _custom_title(
                    "Evidence",
                    margin_top=5,
                    margin_bottom=-20,
                    font_size=25,
                )

                final_answer_context = final_answer.get("final_context", [])
                if len(final_answer_context) > 0:
                    final_answer_context_df = pd.DataFrame(final_answer_context)
                    # st.dataframe(final_answer_context_df)
                    unique_entries_count = (
                        final_answer_context_df["entry_id"].value_counts().to_dict()
                    )

                    images_paths = []
                    shown_str = ""
                    for entry_id, count in unique_entries_count.items():
                        df_one_entry = final_answer_context_df[
                            final_answer_context_df["entry_id"] == entry_id
                        ].iloc[0]
                        extracted_entries = df_one_entry["Extracted Entries"]
                        document_title = df_one_entry["Document Title"]
                        document_publishing_date = df_one_entry[
                            "Document Publishing Date"
                        ]
                        document_source = df_one_entry["Document Source"]
                        if df_one_entry["Entry Type"] in ["PDF Table", "PDF Picture"]:
                            images_paths.append(df_one_entry["entry_fig_path"])

                        shown_str += f"* {extracted_entries} ({document_title}, {document_publishing_date}, {document_source}) - **Number of times mentioned: {count}**\n"

                    st.markdown(shown_str)

                    for image_path in list(set(images_paths)):
                        st.image(image_path)

def _format_func(x):
    return x.split("/")[-1].split(".")[0]

@st.fragment
def main_chat():
    projets_col, _, chat_col = st.columns([0.2, 0.02, 0.78])
    with projets_col:
        projects_selected = st.selectbox(
            "Select a project",
            sorted(list(st.session_state["projects_subprojects_structure"].keys())),
        )
        subprojects_selected = st.selectbox(
            "Select a subproject",
            sorted(
                st.session_state["projects_subprojects_structure"][projects_selected]
            ),
        )
        documents_one_subproject = st.session_state["chat_data"][
            (st.session_state["chat_data"]["project_name"] == projects_selected)
            & (
                st.session_state["chat_data"]["sub_project_name"]
                == subprojects_selected
            )
        ]
        document_names = documents_one_subproject["docs_name"].unique()
        
        chosen_documents = st.multiselect(
            "Select documents",
            document_names,
            format_func=_format_func,
        )
        chat_documents = documents_one_subproject[
            documents_one_subproject["docs_name"].isin(chosen_documents)
        ]
        
        st.markdown(f"#### {projects_selected} - {subprojects_selected}")
        for chosen_document in chosen_documents:
            st.markdown(f"**Document Name**: <a href='{chosen_document}' target='_blank'>{_format_func(chosen_document)}</a>", unsafe_allow_html=True)
            one_doc_df = chat_documents[chat_documents["docs_name"] == chosen_document]
            for col in ["Document Publishing Date", "Document Source", "Document Title"]:
                val = one_doc_df[col].iloc[0]
                if col == "Document Source":
                    val = ", ".join(literal_eval(val))
                st.markdown(f"**{col}**: {val}")
            
            interviewees = one_doc_df["Interviewee"].iloc[0]
            interviewees = literal_eval(interviewees)
            
            if len(interviewees) > 0 and isinstance(interviewees[0], dict):
                st.markdown(f"**Interviewees:**")
                for interviewee in interviewees:
                    name = interviewee["name"]
                    role = interviewee["role"]
                    organization = interviewee["organization"]
                    location = interviewee["location"]
                    st.markdown(f"* **{name}** ({role}, {organization}, {location})".replace(", -", ""))
            else:
                st.markdown(f"**Interviewees Not Found**")
            st.markdown("---")
        


    with chat_col:
        qa_df = chat_documents.drop_duplicates(subset=["Extracted Entries"])
        qa_information_retrieval(qa_df)


main_chat()
