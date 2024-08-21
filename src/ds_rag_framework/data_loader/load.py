import os
import typing as t
from logging import getLogger
from os import getenv
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document, SimpleDirectoryReader
from llama_parse import LlamaParse

from ds_rag_framework.timer import Timer

logger = getLogger(__name__)


class HTMLDataLoader:
    """
    A class for loading HTML documents from URLs and extracting metadata.
    The HTMLDataLoader class provides methods to load HTML documents from a list of URLs
    and extract metadata such as titles and body texts. The loaded documents can be
    further processed and vectorized by the Llama Index.
    Example usage:
    ```
    loader = HTMLDataLoader(input_urls=['https://example.com', 'https://example.org'])
    documents = loader.load_with_simple_text_extraction()
    ```
        Initialize the HTMLDataLoader.
            input_urls (List[str] | str): A list of URLs or a string of URLs separated
                by commas (no space).
            document_version (str, optional): The version of the documents if applicable.
                Defaults to None.
    """
    def __init__(
        self, input_urls: t.List[str] | str, document_version: str = None
    ) -> None:
        """
        Args:
            input_urls: input a list of urls, or a simple string of urls separated
                        by commas (No space)
            docs_version: version of the documents if applicable
        """
        if isinstance(input_urls, str):
            self._input_urls = input_urls.split(",")
        else:
            self._input_urls = input_urls
        self._document_version = document_version if document_version else None

    def load_with_simple_text_extraction(self) -> t.List[Document]:
        """
        Load HTML documents from the given URLs and extract metadata.

        Args:
            input_urls (List[str]): A list of URLs of the HTML documents to load.
            docs_version (str | None, optional): The version of the documents.
                All documents must be from the same version. Defaults to None.

        Returns:
            List[Document]: A list of `Document` objects containing the loaded HTML
                documents and their metadata.
        """

        with Timer() as html_scrape_timer:
            html_contents = [
                requests.get(url, verify=False).content for url in self._input_urls
            ]
            soup_objs = [
                BeautifulSoup(html_content, "html.parser")
                for html_content in html_contents
            ]

            titles = [soup.head.title.string for soup in soup_objs]
            body_texts = [soup.get_text() for soup in soup_objs]
            mod_body_texts = [
                str(body_text).replace("\n", "") for body_text in body_texts
            ]
            mod_body_texts = [
                str(body_text).replace("\r", "") for body_text in mod_body_texts
            ]
        logger.info(
            f"Scraped all the html docs in \
                    {html_scrape_timer.exec_time / 60} mins"
        )

        with Timer() as doc_loader_timer:
            if self._document_version:
                documents = [
                    Document(
                        text=mod_body_text,
                        metadata={
                            "file_name": title,
                            "url": url,
                            "version": self._document_version,
                        },
                    )
                    for (title, url, mod_body_text) in zip(
                        titles, self._input_urls, mod_body_texts
                    )
                ]
            else:
                documents = [
                    Document(
                        text=mod_body_text, metadata={"file_name": title, "url": url}
                    )
                    for (title, url, mod_body_text) in zip(
                        titles, self._input_urls, mod_body_texts
                    )
                ]
        logger.info(
            f"Loaded all the documents in \
                    {doc_loader_timer.exec_time / 60} mins"
        )
        return documents


class PDFDataLoader:
    """
    A class for loading PDF documents and extracting information for vectorization using Llama Index.
    The PDFDataLoader class provides methods to load PDF documents from a directory or a list of files.
    It supports multiple ways of extracting information from PDFs, including using the Llama Parse library
    to parse PDFs into markdown format for better retrieval results, especially for PDFs with tables.
    Usage:
        data_loader = PDFDataLoader(input_dir='/path/to/pdf/files', document_version='v1')
        documents = data_loader.load_with_simple_text_extraction()
        # or
        documents = data_loader.load_with_llama_parse()
    Attributes:
        input_dir (str): The directory path containing the PDF files.
        input_files (List[str]): A list of file paths of the PDF files.
        document_version (str): The version of the document to load.
    Methods:
        load_with_simple_text_extraction(): Load PDF documents using simple text extraction.
        load_with_llama_parse(): Load PDF documents using the Llama Parse library.
        Initialize the PDFDataLoader.
            ValueError: If both `input_dir` and `input_files` are provided.
            ValueError: If the input directory does not exist.
            ValueError: If no valid PDF files found in the input directory.
        ...
        Load PDF documents from a directory or a list of files using simple text extraction.
        ...
        Load PDF documents from a directory or a list of files using the Llama Parse library.
            ValueError: If the LLAMA_CLOUD_API_KEY environment variable is not set.
        ...
    """
    def __init__(
        self,
        input_dir: str | None = None,
        input_files: list | None = None,
        document_version: str | None = None,
    ) -> None:
        """
        Args:
            input_dir (str, optional): The directory path containing the PDF files.
                Defaults to None.
            input_files (List[str], optional): A list of file paths of the PDF files.
                Defaults to None.
            document_version (str, optional): The version of the document to load.
                Defaults to None.
        """
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
            valid_files = [
                str(file_path) for file_path in Path(input_dir).glob("*.pdf")
            ]
            if not valid_files:
                raise ValueError("No valid PDF files found in the input directory.")
            self._input_files = valid_files
        else:
            self._input_files = input_files if input_files else []
        self._input_dir = input_dir
        self._document_version = document_version if document_version else None

    def load_with_simple_text_extraction(self) -> t.Any:
        """
        Load PDF documents from a directory or a list of files.

        Returns:
            Any: A collection of loaded documents.

        Raises:
            ValueError: If both `input_dir` and `input_files` are None.
        """
        with Timer() as doc_loader_timer:
            if self._document_version:
                documents = SimpleDirectoryReader(
                    input_dir=self._input_dir,
                    input_files=self._input_files
                    if len(self._input_files) > 0
                    else None,
                    file_metadata=lambda filename: {
                        "file_name": filename,
                        "version": self._document_version,
                    },
                ).load_data()
            else:
                documents = SimpleDirectoryReader(
                    input_dir=self._input_dir,
                    input_files=self._input_files
                    if len(self._input_files) > 0
                    else None,
                    file_metadata=lambda filename: {"file_name": filename},
                ).load_data()
        logger.info(
            f"Loaded all the documents in \
                    {doc_loader_timer.exec_time / 60} mins"
        )
        return documents

    def load_with_llama_parse(self) -> t.Any:
        """
        Load PDF documents from a directory or a list of files
        using the LLAMA parser.

        Args:

        Returns:
            Any: A collection of loaded documents.

        Raises:
            ValueError: If both `input_dir` and `input_files` are None.
        """
        if getenv("LLAMA_CLOUD_API_KEY") is None:
            raise ValueError(
                "Please set the LLAMA_CLOUD_API_KEY \
                             environment variable to use LlamaParse."
            )

        with Timer() as doc_loader_timer:
            markdown_parser = LlamaParse(
                api_key=getenv("LLAMA_CLOUD_API_KEY"),
                ignore_errors=False,
                invalidate_cache=True,
                result_type="markdown",
            )
            documents = []
            if self._document_version:
                for input_file in self._input_files:
                    document = markdown_parser.load_data(
                        input_file,
                        extra_info={
                            "file_name": input_file,
                            "version": self._document_version,
                        },
                    )
                    documents += document
            else:
                for input_file in self._input_files:
                    document = markdown_parser.load_data(
                        input_file, extra_info={"file_name": input_file}
                    )
                    documents += document
        logger.info(
            f"Loaded all the documents in \
                    {doc_loader_timer.exec_time / 60} mins"
        )
        return documents


class CSVDataLoader:
    """
    Unfinished class as csv files parsed by llama index turns each row into a document, but rows may be interrelated
    so there could be better ways of handling csv documents, such as the text2pandas library in llama index or graph
    database creation also with llama index support and neo4j graph database.
    """
    def __init__(
            self,
            input_dir: str | None = None,
            input_files: list | None = None,
            document_version: str | None = None,
        ) -> None:
            """
            Args:
                input_dir (str, optional): The directory path containing the PDF files.
                    Defaults to None.
                input_files (List[str], optional): A list of file paths of the PDF files.
                    Defaults to None.
                document_version (str, optional): The version of the document to load.
                    Defaults to None.
            """
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
                valid_files = [
                    str(file_path) for file_path in Path(input_dir).glob("*.csv")
                ]
                if not valid_files:
                    raise ValueError("No valid CSV files found in the input directory.")
                self._input_files = valid_files
            else:
                self._input_files = input_files if input_files else []
            self._input_dir = input_dir
            self._document_version = document_version if document_version else None

    def load_with_simple_text_extraction(self) -> t.Any:
        """
        Load PDF documents from a directory or a list of files.

        Returns:
            Any: A collection of loaded documents.

        Raises:
            ValueError: If both `input_dir` and `input_files` are None.
        """
        with Timer() as doc_loader_timer:
            if self._document_version:
                documents = SimpleDirectoryReader(
                    input_dir=self._input_dir,
                    input_files=self._input_files
                    if len(self._input_files) > 0
                    else None,
                    file_metadata=lambda filename: {
                        "file_name": filename,
                        "version": self._document_version,
                    },
                ).load_data()
            else:
                documents = SimpleDirectoryReader(
                    input_dir=self._input_dir,
                    input_files=self._input_files
                    if len(self._input_files) > 0
                    else None,
                    file_metadata=lambda filename: {"file_name": filename},
                ).load_data()
        logger.info(
            f"Loaded all the documents in \
                    {doc_loader_timer.exec_time / 60} mins"
        )
        return documents