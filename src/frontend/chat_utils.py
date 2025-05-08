import streamlit as st
import pandas as pd
from src.frontend.utils import _custom_title


def _display_chat_answer(final_answer):
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
