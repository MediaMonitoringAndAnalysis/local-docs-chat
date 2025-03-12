import streamlit as st


def _get_projects_subprojects_structure():
    final_structure = {}
    projects = st.session_state["chat_data"]["project_name"].unique()
    for project in projects:
        subprojects = st.session_state["chat_data"][st.session_state["chat_data"]["project_name"] == project]["sub_project_name"].unique()
        final_structure[project] = subprojects
    return final_structure

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
