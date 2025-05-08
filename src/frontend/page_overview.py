import pandas as pd
import streamlit as st


@st.fragment
def display_overview(documents_one_subproject: pd.DataFrame):
    st.markdown("### Documents Overview")

    df = documents_one_subproject.copy().reset_index(drop=True)

    
    df = df[st.session_state["relevant_analysis_cols"]].drop_duplicates(subset=["Document Title"]).reset_index(drop=True)
    df.index = df.index + 1
    
    # df["Document Source"] = df["Document Source"].apply(lambda x: ", ".join(x))
    st.write(
        df.to_html(escape=False, justify='left'),
        unsafe_allow_html=True,
    )
    