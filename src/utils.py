from collections import defaultdict
from typing import List, Dict


def _get_tags_dict(classification_tags: List[str]) -> Dict[str, Dict[str, List[str]]]:
    tags_dict = defaultdict(lambda: defaultdict(list))
    for tag in classification_tags:
        pillar, subpillar, indicator = tag.split("->")
        tags_dict[pillar][subpillar].append(indicator)
    return tags_dict
