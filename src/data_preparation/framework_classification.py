import os
import json
from zero_shot_classification import MultiStepZeroShotClassifier
from typing import List, Dict
import pandas as pd


def _load_json_file(path: os.PathLike) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    
def _get_tags_list(framework_data: Dict[str, Dict[str, List[str]]]) -> List[str]:
    
    tags = []
    
    for pillar, pillar_data in framework_data.items():
        for subpillar, subpillar_data in pillar_data.items():
            for tag in subpillar_data:

                tags.append(f"{pillar}->{subpillar}->{tag}")
                
    return tags


def apply_framework_classification(
    entries: List[str],
    framework_path: os.PathLike,
    additional_questions_path: os.PathLike = None,
) -> List[Dict[str, float]]:
    """
    Apply framework classification to a list of entries.
    
    Args:
        entries: List[str] - The list of entries to classify.
        framework_path: os.PathLike - The path to the framework data.
        additional_questions_path: os.PathLike = None - The path to the additional questions data.
        
    Returns:
        List[Dict[str, float]] - The list of classification outputs.
    """
    
    framework_data = _load_json_file(framework_path)
    if additional_questions_path is not None:
        additional_questions_data = _load_json_file(additional_questions_path)
        framework_data["general_questions"] = additional_questions_data
        
    framework_tags = _get_tags_list(framework_data)
    
    final_tag_to_all_tags = {"->".join(tag.split("->")[1:]): tag for tag in framework_tags}
    with open(os.path.join(os.path.dirname(framework_path), "final_tag_to_all_tags.json"), "w") as f:
        json.dump(final_tag_to_all_tags, f)

    classifier = MultiStepZeroShotClassifier(
        # second_pass_model="gemma3:4b-it-q4_K_M",
        # second_pass_api_key=None,
        # second_pass_pipeline="Ollama",
        do_second_pass=False,
        first_pass_threshold=None
    )
    
    outputs: List[Dict[str, float]] = classifier(
        entries=entries,
        tags=list(final_tag_to_all_tags.keys()),
    )
    
    outputs = [
        {
            final_tag_to_all_tags[t]: p for t, p in output.items()
        }
        for output in outputs
    ]
    
    return outputs
