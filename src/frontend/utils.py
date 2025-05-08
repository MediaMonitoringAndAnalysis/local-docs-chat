import streamlit as st
import hashlib

def _get_projects_subprojects_structure():
    final_structure = {}
    projects = st.session_state["chat_data"]["project_name"].unique()
    for project in projects:
        subprojects = st.session_state["chat_data"][st.session_state["chat_data"]["project_name"] == project]["sub_project_name"].unique()
        final_structure[project] = subprojects
    # st.write(final_structure)
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

def _add_blank_space(n_empty_lines: int):
    for _ in range(n_empty_lines):
        st.write("")
        

def _format_interviewee(interviewees):
    if len(interviewees) > 0 and isinstance(interviewees[0], dict):
        output = ""
        for interviewee in interviewees:
            name = interviewee["name"]
            role = interviewee["role"]
            organization = interviewee["organization"]
            location = interviewee["location"]
            location_type = interviewee["location_type"]
            gender = interviewee["gender"]
            organization_type = interviewee["organisation_type"]
            output += f"{name} ({gender}, {role}, {organization}, {organization_type}, {location_type}, {location})".replace(", -", "")
            output += " // "
        output = output[:-4]
        return output
    else:
        return "**Interviewees Not Found**"
    
def _compress_string(original_str: str) -> str:
    # Compress the string
    # Create a SHA-1 hash object
    hash_object = hashlib.sha1()
    hash_object.update(original_str.encode("utf-8"))

    # Get the hex digest and truncate it
    hashed_string = hash_object.hexdigest()[:16]  # Truncate to 16 characters
    return hashed_string