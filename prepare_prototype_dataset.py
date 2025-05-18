import os
import argparse
import pandas as pd
from src.data_preparation.main_data_prep import main_data_prep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder_path", type=str, default="data/docs/Individual Interview notes")
    parser.add_argument("--output_folder_path", type=str, default="data/dataset_prototype")
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
    parser.add_argument(
        "--task",
        type=str,
        default="protoype",
    )

    args = parser.parse_args()
    sample_bool = args.sample_bool == "true"

    if sample_bool:
        output_folder_path = os.path.join(args.output_folder_path, "sample")
    else:
        output_folder_path = args.output_folder_path

    main_data_prep(
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
        args.task,
    )
