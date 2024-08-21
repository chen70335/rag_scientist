import json
import os
from typing import Optional
from pathlib import Path

from llama_index.core.vector_stores.types import (FilterOperator,
                                                  MetadataFilter,
                                                  MetadataFilters)


def filter_version_at_retrieval(version: str | None) -> MetadataFilters:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="version",
                value=version,
                operator=FilterOperator.EQ
            )
        ]
    )

    return filters


def load_config_json(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def check_env_var(
    env_var: str,
    error_message: Optional[str] = None,
) -> None:
    if os.getenv(env_var) is None:
        raise ValueError(
            f"{error_message}" if error_message
            else f"Please provide an .env variable named {env_var}."
        )
    
def check_input_file_or_dir(
    document_type: str,
    input_dir: str | None = None,
    input_files: list | None = None
):
    if input_dir is None and input_files is None:
        raise ValueError("Both `document_dir` and `document_path` can't be null.")
    if input_dir and input_files:
        raise ValueError(
            "Please provide either `document_dir` or `document_path`,\
                        not both."
        )
    if input_dir:
        if not os.path.exists(input_dir):
            raise ValueError("Input directory does not exist.")
        end_path = "*." + document_type
        valid_files = [
            str(file_path) for file_path in Path(input_dir).glob(end_path)
        ]
        if not valid_files:
            raise ValueError("No valid CSV files found in the input directory.")
        self._input_files = valid_files
    else:
        self._input_files = input_files if input_files else []