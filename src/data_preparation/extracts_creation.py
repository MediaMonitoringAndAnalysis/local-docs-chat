import pandas as pd
from entry_extraction import SemanticEntriesExtractor




def apply_entry_extraction(
    input_data: pd.DataFrame, raw_text_column: str, entries_column: str
) -> pd.DataFrame:
    entries_extractor = SemanticEntriesExtractor()
    entries = entries_extractor(input_data[raw_text_column].tolist())
    input_data[entries_column] = entries
    
    input_data = input_data.explode(entries_column)
    input_data = input_data[input_data[entries_column].str.len() > 5]

    return input_data
