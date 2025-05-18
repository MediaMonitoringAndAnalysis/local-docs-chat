import os
import argparse
import pandas as pd
from src.data_preparation.main_data_prep import main_data_prep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_folder_path", type=str, default="data/docs/All interviews")
    parser.add_argument("--output_folder_path", type=str, default="data/dataset_all_docs_run")
    parser.add_argument("--output_file_name", type=str, default="documents_dataset.csv")
    parser.add_argument("--entries_file_name", type=str, default="entries_dataset.csv")
    parser.add_argument("--analyses_file_name", type=str, required=False)
    parser.add_argument("--raw_text_column", type=str, default="text")
    parser.add_argument("--entries_column", type=str, default="Extracted Entries")
    parser.add_argument("--embeddings_column", type=str, required=False)
    parser.add_argument("--secondary_tags_column", type=str, required=False)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--inference_pipeline", type=str, required=False)
    parser.add_argument("--api_key", type=str, required=False)
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
        required=False,
    )
    parser.add_argument(
        "--framework_classification_column",
        type=str,
        default="Framework Probabilities",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="general",
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
        args.embeddings_column if args.embeddings_column else None,
        args.secondary_tags_column if args.secondary_tags_column else None,
        args.framework_classification_column if args.framework_classification_column else None,
        args.framework_path if args.framework_path else None,
        args.additional_questions_path if args.additional_questions_path else None,
        args.analyses_file_name if args.analyses_file_name else None,
        args.model_name if args.model_name else None,
        args.inference_pipeline if args.inference_pipeline else None,
        args.api_key if args.api_key else None,
        sample_bool,
        args.task,
    )
