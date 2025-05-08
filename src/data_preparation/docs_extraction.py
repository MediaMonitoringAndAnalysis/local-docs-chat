import os
import sys
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from documents_processing import DocumentsDataExtractor, supported_file_extensions


def _extract_docs_data(
    documents_raw_data_extractor,
    final_data: pd.DataFrame,
    to_be_extrcted_docs: Dict[str, Dict[str, List[os.PathLike]]],
    extracted_docs_names: List[os.PathLike],
    output_path: os.PathLike,
):
    """
    Extract data from documents
    """

    n_to_be_extrcted_docs = 0
    for project_name, sub_projects in to_be_extrcted_docs.items():
        for sub_project_name, docs_paths_sub_project in sub_projects.items():
            n_to_be_extrcted_docs += len(docs_paths_sub_project)

    if n_to_be_extrcted_docs > 0:
        tqdm_bar = tqdm(total=n_to_be_extrcted_docs, desc="Extracting documents text")
        print(
            f"##################### Extracting {n_to_be_extrcted_docs} documents #####################"
        )

        for project_name, sub_projects_docs in to_be_extrcted_docs.items():
            for sub_project_name, docs_paths_sub_project in sub_projects_docs.items():
                for doc_path in docs_paths_sub_project:
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
                            metadata_extraction_type="interview",
                            extract_figures_bool=True,
                            relevant_pages_for_metadata_extraction=[0]
                        )
                        data_sub_project["project_name"] = project_name
                        data_sub_project["sub_project_name"] = sub_project_name
                        data_sub_project["docs_name"] = doc_path
                        final_data = pd.concat([final_data, data_sub_project])
                        extracted_docs_names.append(os.path.basename(doc_path))
                    except Exception as e:
                        print(f"Error extracting data from {doc_file_name}: {e}")
                        
                    tqdm_bar.update(1)

                    final_data.to_csv(output_path, index=False)

    final_data["doc_id"] = [i for i in range(len(final_data))]

    final_data.to_csv(output_path, index=False)

    return final_data


def extract_documents_text(
    docs_folder_path: os.PathLike,
    output_path: os.PathLike,
    model_name: str,
    inference_pipeline: str,
    api_key: str,
    sample_bool: bool,
):

    if os.path.exists(output_path):
        final_data = pd.read_csv(output_path)
        extracted_docs_names = final_data["docs_name"].tolist()
        extracted_docs_names = [
            os.path.basename(doc_name) for doc_name in extracted_docs_names
        ]
        # print(f"Extracted docs names: {extracted_docs_names}")
    else:
        final_data = pd.DataFrame()
        extracted_docs_names = []

    projects_names = os.listdir(docs_folder_path)

    documents_raw_data_extractor = DocumentsDataExtractor(
        inference_pipeline_name=inference_pipeline,
        model_name=model_name,
        api_key=api_key
    )

    to_be_extrcted_docs = defaultdict(lambda: defaultdict(list))
    for project_name in projects_names:
        project_path = os.path.join(docs_folder_path, project_name)
        if os.path.isdir(project_path):
            sub_projects_names = os.listdir(project_path)
            for sub_project_name in sub_projects_names:
                sub_project_path = os.path.join(project_path, sub_project_name)
                if os.path.isdir(sub_project_path):
                    docs_paths_sub_project = [
                        os.path.join(sub_project_path, f)
                        for f in os.listdir(sub_project_path)
                        if os.path.isfile(os.path.join(sub_project_path, f))
                        and os.path.splitext(f)[1] in supported_file_extensions
                        and f not in extracted_docs_names
                    ]
                    if sample_bool:
                        docs_paths_sub_project = docs_paths_sub_project[:2]
                    # print(f"Docs paths sub project: {docs_paths_sub_project}")
                    to_be_extrcted_docs[project_name][sub_project_name].extend(
                        docs_paths_sub_project
                    )

    _extract_docs_data(
        documents_raw_data_extractor,
        final_data,
        to_be_extrcted_docs,
        extracted_docs_names,
        output_path,
    )
