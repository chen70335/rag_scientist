import os
import sys
from typing import List, Literal, Optional

from llama_index.core import Document

from ds_rag_framework.data_loader.load import HTMLDataLoader, PDFDataLoader
from ds_rag_framework.llms.llm_picker import LLMPicker
from ds_rag_framework.utils import load_config_json
from ds_rag_framework.vectorizer.vectorizer import DocVectorizer


# TODO: refactor create_vector_store function to allow for easier understanding and input checks
class VectorStoreCreator:
    """
    A class that allows users to extract information from documents in different ways and vectorize them to either local, azure, or vertex.
    This class depends on the data_loader classes to load and extract text from documents. It provides methods to create a vector store for a given document type and store it in the specified location.
    Methods:
    - create_vector_store(): Creates a vector store for the given document type and stores it in the specified location.
    - _input_check(): Performs input validation for the create_vector_store() method.
    - _load_documents(): Loads and extracts text from the input documents using the appropriate data_loader.
    - _create_local_vector_store(): Creates a local vector store using the provided documents and parameters.
    - _create_cog_search_vector_store(): Creates a vector store in Azure Cognitive Search using the provided documents and parameters.
    """
    def create_vector_store(
        self,
        document_type: str,
        index_path: str,
        location: Literal["local", "cog_search", "vertex_ai"],
        vertex_index_endpoint: Optional[str] = None,
        extraction_method: str = "simple",
        input_urls: Optional[list[str]] = None,
        input_dir: Optional[str] = None,
        docs_version: Optional[str] = None,
        filterable_metadata_fields: Optional[list[str]] = None,
        merge: bool = False,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Creates a vector store for the given document type and stores it in the specified location.
        This class is dependent on the DocVectorizer class in ../vectorizer/vectorizer.py.
        This class is meant to be used by a client, while the DocVectorizer represents the underlying logic for vectorizing documents.

        Parameters:
        - document_type (str): The type of document to vectorize.
        - index_path (str): The path where the vector store will be stored.
        - location (Literal['local', 'cog_search', 'vertex_ai']): The location where the vector store will be stored.
          Must be one of 'local', 'cog_search', or 'vertex_ai'.
        - index_endpoint (Optional[str]): The endpoint for the Google Vertex index. This parameter is only applicable
          when the location is set to 'vertex_ai'.
        - extraction_method (str): The method used for extracting text from the documents. Default is 'simple'.
        - input_urls (Optional[list[str]]): A list of URLs pointing to the input documents. This parameter is only
          applicable when the extraction_method is set to 'web'.
        - input_dir (Optional[str]): The directory path containing the input documents. This parameter is only applicable
          when the extraction_method is set to 'local'.
        - docs_version (Optional[str]): The version of the documents. Default is None.
        - merge (bool): Whether to merge the vector store with an existing one at the specified index_path. Default is False.
        - llm_model (Optional[str]): The language model to use for document embedding. Default is None.
        - embed_model (Optional[str]): The embedding model to use for document embedding. Default is None.
        - chunk_size (Optional[int]): The number of documents to process in each chunk. Default is None.

        Raises:
        - ValueError: If the location is not one of 'local', 'cog_search', or 'vertex_ai'.

        Returns:
        - None
        """
        if location not in ["local", "cog_search", "vertex_ai"]:
            raise ValueError(
                "Please provide a location to store the vector store. Specify either local, cog_search, or vertex_ai"
            )
        self._input_check(
            document_type=document_type,
            extraction_method=extraction_method,
            input_urls=input_urls,
            input_dir=input_dir,
        )
        documents = self._load_documents(
            document_type=document_type,
            extraction_method=extraction_method,
            input_urls=input_urls,
            input_dir=input_dir,
            docs_version=docs_version,  # TODO: allow more metadata fields other than version
        )
        if location == "local":
            self._create_local_vector_store(
                documents=documents,
                index_path=index_path,
                merge=merge,
                llm_model=llm_model,
                embed_model=embed_model,
                chunk_size=chunk_size,
            )
        elif location == "cog_search":
            self._create_cog_search_vector_store(
                documents=documents,
                index_path=index_path,
                filterable_metadata_fields=filterable_metadata_fields,
                merge=merge,
                llm_model=llm_model,
                embed_model=embed_model,
                chunk_size=chunk_size,
            )
        elif location == 'vertex_ai':
            # TODO: test vertex_ai vector store creation
            self._create_vertex_vector_store(
                documents=documents,
                index_path=index_path,
                index_endpoint=index_path,
                docs_version=docs_version,
                merge=merge,
                llm_model=llm_model,
                embed_model=embed_model,
                chunk_size=chunk_size
            )

    def _input_check(
        self,
        document_type: str,
        extraction_method: str,
        input_urls: Optional[list[str]] = None,
        input_dir: Optional[str] = None,
    ) -> None:
        if document_type not in ["html", "pdf"]:
            raise ValueError("Document type must be either 'html' or 'pdf'.")
        if input_urls is None and input_dir is None:
            raise ValueError("Please provide either input_urls or input_dir.")
        if input_dir is not None and not os.path.exists(input_dir):
            raise ValueError("Input directory does not exist.")

        if extraction_method not in ["simple", "llama_parse"]:
            raise ValueError(
                "Extraction method must be either 'simple' or 'llama_parse' for pdfs. only 'simple' for html"
            )

    def _load_documents(
        self,
        document_type: str,
        extraction_method: str,
        input_urls: Optional[list[str]] = None,
        input_dir: Optional[str] = None,
        docs_version: Optional[str] = None,
    ) -> List[Document]:
        data_loader = (
            HTMLDataLoader(input_urls=input_urls, document_version=docs_version)
            if document_type == "html"
            else PDFDataLoader(input_dir=input_dir, document_version=docs_version)
        )
        documents = None
        if document_type == "html":
            documents = data_loader.load_with_simple_text_extraction()
        else:
            if extraction_method == "simple":
                documents = data_loader.load_with_simple_text_extraction()
            else:
                documents = data_loader.load_with_llama_parse()
        return documents

    def _create_local_vector_store(
        self,
        documents: List[Document],
        index_path: str,
        merge: bool = False,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Initializes the VectorStoreCreator object.

        Args:
            document_type (str): The type of document to vectorize. Must be either 'html' or 'pdf'.
            index_path (str): The path to store the vector store.
            extraction_method (str, optional): The method to extract text from documents. Defaults to 'simple'.
            input_urls (list[str], optional): The list of URLs of the input documents. Defaults to None.
            input_dir (str, optional): The directory path of the input documents. Defaults to None.
            docs_version (str, optional): The version of the documents. Defaults to None.
            merge (bool, optional): Whether to merge the vector store with an existing one. Defaults to False.
            llm_model (str, optional): The LLM model to use for vectorization. Defaults to None.
            embed_model (str, optional): The embedding model to use for vectorization. Defaults to None.
            chunk_size (int, optional): The chunk size for vectorization. Defaults to None.

        Returns:
            None
        """
        llm_picker = LLMPicker(
            llm_provider=llm_model, embed_model_name=embed_model, chunk_size=chunk_size
        )
        vectorizer = DocVectorizer(llm_picker=llm_picker)
        vectorizer.build_and_store_local(
            documents=documents, index_path=index_path, merge=merge
        )

    def _create_cog_search_vector_store(
        self,
        documents: List[Document],
        index_path: str,
        filterable_metadata_fields: Optional[list[str]] = None,
        merge: bool = False,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        llm_picker = LLMPicker(
            llm_provider=llm_model, embed_model_name=embed_model, chunk_size=chunk_size
        )
        vectorizer = DocVectorizer(llm_picker=llm_picker)
        vectorizer.build_and_store_azure(
            documents=documents,
            filterable_metadata_fields=filterable_metadata_fields,
            index_name=index_path,
            merge=merge,
        )
    def _create_vertex_vector_store(
        self,
        documents: List[Document],
        index_path: str,
        index_endpoint : str,
        docs_version: Optional[str] = None,
        merge: bool = False,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        # TODO: test vertext AI integration
        llm_picker = LLMPicker(llm_provider=llm_model, embed_model_name=embed_model, chunk_size=chunk_size)
        vectorizer = DocVectorizer(llm_picker=llm_picker)
        vectorizer.build_and_store_vertex(
            documents=documents,
            documents_version=docs_version if docs_version else None,
            index_name=index_path,
            vertex_index_endpoint=index_endpoint,
            merge=merge
        )

if __name__ == "__main__":
    """
    Main function to create a vector store based on the provided configuration file.
    """
    if len(sys.argv) != 2:
        raise ValueError(
            "Please provide the path to the config file. Ex: python src/ds_rag_framework/agents/vectorize.py src/ds_rag_framework/configs/vectorize_config.json"
        )
    config_file_path = sys.argv[1]
    config = load_config_json(config_file_path)
    params = config["vector_store"] | config["llm_settings"]
    VectorStoreCreator().create_vector_store(**params)
