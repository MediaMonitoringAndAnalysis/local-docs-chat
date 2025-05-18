import pandas as pd
from entry_extraction import SemanticEntriesExtractor
from typing import List, Union, Dict



def apply_entry_extraction(
    input_data: pd.DataFrame, raw_text_column: str, entries_column: str, delete_question_entries: bool = True
) -> pd.DataFrame:
    entries_extractor = SemanticEntriesExtractor(delete_question_entries=delete_question_entries)
    entries: List[List[Union[Dict[str, Union[str, int]], str]]] = entries_extractor(input_data[raw_text_column].tolist())

    input_data[entries_column] = entries
    
    # Check if entries are dictionaries and split into separate columns if needed
    if entries and isinstance(entries[0][0], dict):
        # Create separate columns for text and page number
        input_data['text'] = input_data[entries_column].apply(lambda x: [entry['text'] for entry in x])
        input_data['page_number'] = input_data[entries_column].apply(lambda x: [entry['page'] for entry in x])
        # Drop the original entries column
        input_data = input_data.drop(columns=[entries_column])
        # Explode both columns
        input_data = input_data.explode(['text', 'page_number']).rename(columns={'text': entries_column})
    else:
        # Original behavior for non-dictionary entries
        input_data = input_data.explode(entries_column)
        
    # Filter out entries with text length <= 5
    input_data = input_data[input_data[entries_column].str.len() > 5]

    return input_data
