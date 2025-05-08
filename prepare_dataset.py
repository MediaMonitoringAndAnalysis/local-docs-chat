import os
import argparse
import pandas as pd
from src.data_preparation.docs_extraction import (
    extract_documents_text,
)
from src.data_preparation.embeddings import _add_embeddings_column
from src.data_preparation.extracts_creation import apply_entry_extraction
from src.data_preparation.secondary_tags_classification import generate_secondary_tags
from src.data_preparation.framework_classification import apply_framework_classification
from src.data_preparation.generate_analyses import generate_analyses

def main(
    docs_folder_path: str,
    output_folder_path: str,
    documents_file_name: str,
    entries_file_name: str,
    raw_text_column: str,
    entries_column: str,
    embeddings_column: str,
    secondary_tags_column: str,
    framework_classification_column: str,
    framework_path: str,
    additional_questions_path: str,
    analyses_file_name: str,
    model_name: str,
    inference_pipeline: str,
    api_key: str,
    sample_bool: bool,
):

    os.makedirs(output_folder_path, exist_ok=True)

    documents_output_path = os.path.join(output_folder_path, documents_file_name)
    entries_output_path = os.path.join(output_folder_path, entries_file_name)
    analyses_output_path = os.path.join(output_folder_path, analyses_file_name)
    if not os.path.exists(documents_output_path):
        extract_documents_text(
            docs_folder_path,
            documents_output_path,
            model_name,
            inference_pipeline,
            api_key,
            sample_bool
        )

    if not os.path.exists(entries_output_path):
        documents_raw_data = pd.read_csv(documents_output_path)
        entries_data = apply_entry_extraction(
            documents_raw_data,
            raw_text_column=raw_text_column,
            entries_column=entries_column,
            delete_question_entries=True,
        )
        entries_data["entry_id"] = [i for i in range(len(entries_data))]
        entries_data = entries_data.drop(columns=[raw_text_column])
        entries_data.to_csv(entries_output_path, index=False)
    else:
        entries_data = pd.read_csv(entries_output_path)

    if embeddings_column not in entries_data.columns:
        entries_data = _add_embeddings_column(
            entries_data,
            text_column=entries_column,
            embeddings_column=embeddings_column,
        )
        entries_data.to_csv(entries_output_path, index=False)

    if secondary_tags_column not in entries_data.columns:
        secondary_tags_data = generate_secondary_tags(
            entries_data[entries_column].tolist(),
        )
        entries_data[secondary_tags_column] = secondary_tags_data
        entries_data.to_csv(entries_output_path, index=False)

    if framework_classification_column not in entries_data.columns or sample_bool:
        zero_shot_outputs = apply_framework_classification(
            entries_data[entries_column].tolist(),
            framework_path=framework_path,
            additional_questions_path=additional_questions_path,
        )
        entries_data[framework_classification_column] = zero_shot_outputs
        entries_data.to_csv(entries_output_path, index=False)

    if not os.path.exists(analyses_output_path):
        generate_analyses(
            entries_output_path,
            analyses_output_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder_path", type=str, default="data/docs")
    parser.add_argument("--output_folder_path", type=str, default="data/dataset")
    parser.add_argument("--output_file_name", type=str, default="documents_dataset.csv")
    parser.add_argument("--entries_file_name", type=str, default="entries_dataset.csv")
    parser.add_argument("--analyses_file_name", type=str, default="analyses_dataset.json")
    parser.add_argument("--raw_text_column", type=str, default="text")
    parser.add_argument("--entries_column", type=str, default="Extracted Entries")
    parser.add_argument("--embeddings_column", type=str, default="Embeddings")
    parser.add_argument("--secondary_tags_column", type=str, default="Secondary Tags")
    parser.add_argument("--model_name", type=str, default="gemma3:12b-it-q4_K_M")
    parser.add_argument("--inference_pipeline", type=str, default="Ollama")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--sample_bool", type=str, default="false")
    parser.add_argument(
        "--framework_path",
        type=str,
        default=os.path.join(
            "data", "frameworks", "prototype", "original_framework.json"
        ),
    )
    parser.add_argument(
        "--additional_questions_path",
        type=str,
        default=os.path.join(
            "data", "frameworks", "prototype", "additional_questions.json"
        ),
    )
    parser.add_argument(
        "--framework_classification_column",
        type=str,
        default="Framework Probabilities",
    )

    args = parser.parse_args()
    sample_bool = args.sample_bool == "true"

    if sample_bool:
        output_folder_path = os.path.join(args.output_folder_path, "sample")
    else:
        output_folder_path = args.output_folder_path

    main(
        args.docs_folder_path,
        output_folder_path,
        args.output_file_name,
        args.entries_file_name,
        args.raw_text_column,
        args.entries_column,
        args.embeddings_column,
        args.secondary_tags_column,
        args.framework_classification_column,
        args.framework_path,
        args.additional_questions_path,
        args.analyses_file_name,
        args.model_name,
        args.inference_pipeline,
        args.api_key,
        sample_bool,
    )
