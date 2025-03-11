import pandas as pd
from embeddings_generator import EmbeddingsGenerator


def _add_embeddings_column(
    input_data: pd.DataFrame, text_column: str, embeddings_column: str
) -> pd.DataFrame:
    # Initialize with default model
    generator = EmbeddingsGenerator()

    # Or specify a model
    generator = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings for a single text
    embeddings = generator(input_data[text_column].tolist())
    input_data[embeddings_column] = embeddings.tolist()
    return input_data