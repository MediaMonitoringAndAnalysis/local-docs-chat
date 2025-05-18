import pandas as pd
import streamlit as st
import extra_streamlit_components as stx
from src.frontend.chat_utils import _display_chat_answer

breakdown_mapping_values = {
    "M": "Male",
    "F": "Female",
    "M, F": "Mixed",
    "F, M": "Mixed",
    "M-F": "Mixed",
}


def _show_most_relevant_data(
    documents_one_subproject: pd.DataFrame,
    final_tag: str,
    breakdown_metadata_selected: str,
    probas_col: str = "Framework Probabilities",
    decision_threshold: float = 0.6,
):
    if breakdown_metadata_selected == "No Breakdown":
        additional_cols = []
    else:
        additional_cols = [breakdown_metadata_selected]
    relevant_cols = [
        "Extracted Entries",
        "Document Title",
        "Document Publishing Date",
        "Document Source",
        "Interviewee",
        # "Number of Pages",
        *additional_cols,
        # "Secondary Tags",
        "Framework Probabilities",
    ]
    df = documents_one_subproject.copy()

    df[probas_col] = df[probas_col].apply(lambda x: x[final_tag])
    df = df[df[probas_col] >= decision_threshold].reset_index(drop=True)
    df = df.sort_values(by=probas_col, ascending=False).reset_index(drop=True)[
        relevant_cols
    ]
    df.index = df.index + 1
    df["Framework Probabilities"] = df["Framework Probabilities"].apply(
        lambda x: f"{100*x:.0f}%"
    )
    if breakdown_metadata_selected == "No Breakdown":
        st.write(df.rename(columns={"Framework Probabilities": "Relevance Score"}).to_html(escape=False, justify="left"), unsafe_allow_html=True)
    else:
        breakdown_values = df[breakdown_metadata_selected].unique()
        for breakdown_value in breakdown_values:
            st.write(
                f"#### {breakdown_mapping_values.get(breakdown_value, breakdown_value)}"
            )
            st.write(
                df[df[breakdown_metadata_selected] == breakdown_value].rename(columns={"Framework Probabilities": "Relevance Score"}).to_html(
                    escape=False, justify="left"
                ),
                unsafe_allow_html=True,
            )

@st.fragment
def _show_analysis(
    project: str,
    subproject: str,
    pillar: str,
    subpillar: str,
    indicator: str,
    breakdown: str,
):
    try:
        analyses_one_breakdown = st.session_state["analyses_data"][project][subproject][pillar][subpillar][indicator][breakdown]
    except:
        analyses_one_breakdown = {"Executive Summary": "...", "By Breakdown": {}}
        
    if len(analyses_one_breakdown['Executive Summary'])>0:
        st.write("#### Executive Summary")
        executive_summary = analyses_one_breakdown['Executive Summary']
        if type(executive_summary) == str:
            executive_summary = executive_summary.split("/think")[-1]
            st.write(executive_summary)
        else:
            _display_chat_answer(executive_summary)
            
    if len(analyses_one_breakdown['By Breakdown'])>0:
        for breakdown, breakdown_value in analyses_one_breakdown['By Breakdown'].items():
            
            st.write(f"#### {breakdown_mapping_values[breakdown]} Analysis")
            # st.write(breakdown_value["Executive Summary"])
            _display_chat_answer(breakdown_value["Executive Summary"])
    


@st.fragment
def _show_results(
    documents_one_subproject: pd.DataFrame,
    final_tag: str,
    breakdown_metadata_selected: str,
    project: str,
    subproject: str,
    pillar: str,
    subpillar: str,
    indicator: str,
    breakdown: str,
):
    # st.write(f"#### Analysis for {final_tag}")
    # st.write(documents_one_subproject)

    tabs_data = [
        "Relevant Data",
        "Analysis",
    ]

    bar_data = [
        stx.TabBarItemData(id=i + 1, title=pillar, description="")
        for i, pillar in enumerate(tabs_data)
    ]

    chosen_id = stx.tab_bar(data=bar_data, default=1, return_type=int)
    tab_name = tabs_data[chosen_id - 1]

    if tab_name == "Relevant Data":
        _show_most_relevant_data(
            documents_one_subproject, final_tag, breakdown_metadata_selected
        )
    elif tab_name == "Analysis":
        _show_analysis(project, subproject, pillar, subpillar, indicator, breakdown)


@st.fragment
def display_analysis(documents_one_subproject: pd.DataFrame, project: str, subproject: str):
    # st.write("### Documents Analysis")

    # with metadata_col:
    st.write("#### Document Metadata")

    pillars = list(st.session_state["tags_dict"].keys())
    pillars_selected = st.selectbox("Pillar", pillars)
    subpillars = list(st.session_state["tags_dict"][pillars_selected].keys())
    subpillars_selected = st.selectbox("Subpillar", subpillars)
    indicators = list(
        st.session_state["tags_dict"][pillars_selected][subpillars_selected]
    )
    indicators_selected = st.selectbox("Indicator", indicators)
    
    final_tag = f"{pillars_selected}->{subpillars_selected}->{indicators_selected}"

    breakdown_metadata = [
        "No Breakdown",
        # "Interviewee Location Type",
        # "Interviewee Organization Type",
        "Interviewee Gender",
    ]
    breakdown_metadata_selected = st.selectbox("Breakdown by", breakdown_metadata)

    if st.button("Generate Analysis", key="generate_analysis_button"):
        _show_results(
            documents_one_subproject,
            final_tag,
            breakdown_metadata_selected,
            project,
            subproject,
            pillars_selected,
            subpillars_selected,
            indicators_selected,
            breakdown_metadata_selected,
        )
