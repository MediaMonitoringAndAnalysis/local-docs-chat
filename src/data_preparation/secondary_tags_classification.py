import os
import sys
from typing import List


def generate_secondary_tags(entries: List[str], prediction_ratio: float = 0.9) -> List[List[str]]:
    """
    Generate secondary tags for a list of entries.
    """
    os.chdir(
        os.path.join(
            "..", "nlp_pipelines", "humanitarian-extract-classificator"
        )
    )
    sys.path.append(os.getcwd())
    
    from main import humbert_classification
    
    humbert_outputs = humbert_classification(
        entries, prediction_ratio=prediction_ratio
    )

    secondary_tags = []
    for one_output in humbert_outputs:
        one_secondary_tags_output = ["->".join(t.split("->")[1:]) for t in one_output if "secondary" in t.lower()]
        secondary_tags.append(one_secondary_tags_output)
        
    os.chdir(
        os.path.join(
            "..", "..", "local-docs-chat"
        )
    )
    sys.path.append(os.getcwd())
    
    return secondary_tags


