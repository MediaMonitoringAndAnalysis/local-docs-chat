import os
import sys
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from documents_processing import DocumentsDataExtractor, supported_file_extensions


def _get_to_be_extracted_docs(
    docs_folder_path: os.PathLike,
    files_terminations: List[str] = supported_file_extensions,
):
    to_be_extracted_docs = []
    for root, dirs, files in os.walk(docs_folder_path):
        for file in files:
            print(file)
            if any(file.lower().endswith(ext) for ext in files_terminations):
                file_path = os.path.join(root, file)
                to_be_extracted_docs.append(file_path)
    return to_be_extracted_docs


def _extract_docs_data(
    documents_raw_data_extractor,
    final_data: pd.DataFrame,
    to_be_extracted_docs: List[os.PathLike],
    extracted_docs_paths: List[os.PathLike],
    output_path: os.PathLike,
):
    """
    Extract data from documents
    """
    doc_name_to_index = {doc_path: i for i, doc_path in enumerate(to_be_extracted_docs)}
    to_be_extracted_docs = [
        doc_path for doc_path in to_be_extracted_docs if doc_path not in extracted_docs_paths
    ]
    n_to_be_extracted_docs = len(to_be_extracted_docs)
    
    if documents_raw_data_extractor.inference_pipeline_name is not None:
        metadata_extraction_type = "interview"
        extract_figures_bool = True
        relevant_pages_for_metadata_extraction = [0]
    else:
        metadata_extraction_type = False
        extract_figures_bool = False
        relevant_pages_for_metadata_extraction = None

    if n_to_be_extracted_docs > 0:
        tqdm_bar = tqdm(total=n_to_be_extracted_docs, desc="Extracting documents text")
        print(
            f"##################### Extracting {n_to_be_extracted_docs} documents #####################"
        )

        for doc_path in to_be_extracted_docs:
            doc_file_path = os.path.abspath(doc_path)
            doc_folder_path = os.path.dirname(doc_file_path)
            doc_file_name = os.path.basename(doc_file_path)
            figures_saving_path = os.path.join(
                doc_folder_path, "..", "..", "figures"
            )
            # print(f"Extracting data from {doc_file_name}")
            # print(f"Doc folder path: {doc_folder_path}")
            try:
                data_sub_project = documents_raw_data_extractor(
                    file_name=doc_file_name,
                    doc_folder_path=doc_folder_path,
                    figures_saving_path=figures_saving_path,
                    metadata_extraction_type=metadata_extraction_type,
                    extract_figures_bool=extract_figures_bool,
                    relevant_pages_for_metadata_extraction=relevant_pages_for_metadata_extraction,
                    return_original_pages_numbers=True
                    )
                data_sub_project["docs_path"] = doc_path
                final_data = pd.concat([final_data, data_sub_project])
                extracted_docs_paths.append(doc_path)
            except Exception as e:
                print(f"Error extracting data from {doc_file_name}: {e}")
                
            tqdm_bar.update(1)

            final_data.to_csv(output_path, index=False)

    final_data["doc_id"] = [doc_name_to_index.get(doc_path, doc_path) for doc_path in final_data["docs_path"]]

    final_data.to_csv(output_path, index=False)

    return final_data


def extract_documents_text_general(
    docs_folder_path: os.PathLike,
    output_path: os.PathLike,
    files_terminations: List[str] = supported_file_extensions,
    model_name: Optional[str] = None,
    inference_pipeline: Optional[str] = None,
    api_key: Optional[str] = None,
    sample_bool: bool = False,
):

    if os.path.exists(output_path):
        final_data = pd.read_csv(output_path)
        extracted_docs_paths = final_data["docs_paths"].tolist()
        # print(f"Extracted docs names: {extracted_docs_names}")
    else:
        final_data = pd.DataFrame()
        extracted_docs_paths = []
        
    documents_raw_data_extractor = DocumentsDataExtractor(
        inference_pipeline_name=inference_pipeline,
        model_name=model_name,
        api_key=api_key
    )

    to_be_extracted_docs = _get_to_be_extracted_docs(docs_folder_path, files_terminations)
    
    # If sample mode is enabled, limit to a small subset
    
    if sample_bool and len(to_be_extracted_docs) > 3:
        to_be_extracted_docs = to_be_extracted_docs[:3]

    _extract_docs_data(
        documents_raw_data_extractor,
        final_data,
        to_be_extracted_docs,
        extracted_docs_paths,
        output_path,
    )
