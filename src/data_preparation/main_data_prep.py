import os
import pandas as pd
from typing import Optional
from src.data_preparation.docs_extraction_protoype import (
    extract_documents_text_protoype,
)
from src.data_preparation.docs_extraction_general import extract_documents_text_general
from src.data_preparation.embeddings import _add_embeddings_column
from src.data_preparation.extracts_creation import apply_entry_extraction
from src.data_preparation.secondary_tags_classification import generate_secondary_tags
from src.data_preparation.framework_classification import apply_framework_classification
from src.data_preparation.generate_analyses import generate_analyses
from ast import literal_eval

task_to_extraction_function = {
    "protoype": extract_documents_text_protoype,
    "general": extract_documents_text_general,
}

def _custom_eval(x):
    try:
        return literal_eval(x)
    except:
        return x

def main_data_prep(
    docs_folder_path: os.PathLike,
    output_folder_path: os.PathLike,
    documents_file_name: str,
    entries_file_name: str,
    raw_text_column: str,
    entries_column: str,
    embeddings_column: Optional[str] = None,
    secondary_tags_column: Optional[str] = None,
    framework_classification_column: Optional[str] = None,
    framework_path: Optional[os.PathLike] = None,
    additional_questions_path: Optional[os.PathLike] = None,
    analyses_file_name: Optional[str] = None,
    model_name: Optional[str] = None,
    inference_pipeline: Optional[str] = None,
    api_key: Optional[str] = None,
    sample_bool: bool = False,
    task: str = "general",
):

    os.makedirs(output_folder_path, exist_ok=True)

    documents_output_path = os.path.join(output_folder_path, documents_file_name)
    entries_output_path = os.path.join(output_folder_path, entries_file_name)
    if analyses_file_name:
        analyses_output_path = os.path.join(output_folder_path, analyses_file_name)
    else:
        analyses_output_path = None
    if not os.path.exists(documents_output_path):
        if task == "protoype":
            extract_documents_text_protoype(
                docs_folder_path,
                documents_output_path,
                model_name=model_name,
                inference_pipeline=inference_pipeline,
                api_key=api_key,
                sample_bool=sample_bool
            )
        elif task == "general":
            extract_documents_text_general(
                docs_folder_path,
                documents_output_path,
                model_name=model_name,
                inference_pipeline=inference_pipeline,
                api_key=api_key,
                sample_bool=sample_bool
            )

    if not os.path.exists(entries_output_path):
        documents_raw_data = pd.read_csv(documents_output_path)
        documents_raw_data[raw_text_column] = documents_raw_data[raw_text_column].apply(_custom_eval)
        entries_data = apply_entry_extraction(
            documents_raw_data,
            raw_text_column=raw_text_column,
            entries_column=entries_column,
            delete_question_entries=True,
        )
        entries_data["entry_id"] = [i for i in range(len(entries_data))]
        # entries_data = entries_data.drop(columns=[raw_text_column])
        entries_data.to_csv(entries_output_path, index=False)
    else:
        entries_data = pd.read_csv(entries_output_path)
        
    if sample_bool:
        entries_data = entries_data.sample(n=15)

    if embeddings_column and (embeddings_column not in entries_data.columns or sample_bool):
        entries_data = _add_embeddings_column(
            entries_data,
            text_column=entries_column,
            embeddings_column=embeddings_column,
        )
        entries_data.to_csv(entries_output_path, index=False)

    if secondary_tags_column and (secondary_tags_column not in entries_data.columns or sample_bool):
        secondary_tags_data = generate_secondary_tags(
            entries_data[entries_column].tolist(),
        )
        entries_data[secondary_tags_column] = secondary_tags_data
        entries_data.to_csv(entries_output_path, index=False)

    if framework_classification_column and (framework_classification_column not in entries_data.columns or sample_bool):
        zero_shot_outputs = apply_framework_classification(
            entries_data[entries_column].tolist(),
            framework_path=framework_path,
            additional_questions_path=additional_questions_path,
        )
        entries_data[framework_classification_column] = zero_shot_outputs
        entries_data.to_csv(entries_output_path, index=False)

    if analyses_file_name and not os.path.exists(analyses_output_path):
        generate_analyses(
            entries_output_path,
            analyses_output_path,
        )
