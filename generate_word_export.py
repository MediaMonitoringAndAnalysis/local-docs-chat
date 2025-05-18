import os
import argparse
from src.data_preparation.export_results_to_word import generate_evidences_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_folder_path", type=str, default="data/dataset_all_docs_run/sample")
    parser.add_argument("--entries_dataset_name", type=str, default="entries_dataset.csv")
    parser.add_argument("--framework_file_path", type=str, default="data/frameworks/prototype/original_framework.json")
    parser.add_argument("--output_file_name", type=str, default="evidences_doc.docx")
    parser.add_argument("--cutoff_threshold", type=float, default=0.6)
    parser.add_argument("--doc_title", type=str, default="All Interviews Evidences")
    args = parser.parse_args()
    
    analysis_file_path = os.path.join(args.outputs_folder_path, args.entries_dataset_name)
    output_file_path = os.path.join(args.outputs_folder_path, args.output_file_name)

    generate_evidences_doc(
        analysis_file_path, args.framework_file_path, output_file_path, args.doc_title, args.cutoff_threshold
    )
