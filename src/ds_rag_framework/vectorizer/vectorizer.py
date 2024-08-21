import os
import typing as t
from logging import getLogger

from llama_index.core import (Document, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser

from ds_rag_framework.llms.llm_picker import LLMPicker
from ds_rag_framework.timer import Timer
from ds_rag_framework.vector_stores.cogsearch import CogSearchClient
from ds_rag_framework.vector_stores.vertexai import VertexAIClient  

logger = getLogger(__name__)


# TODO: add transformation functionality here and be able to customize splitter
class DocVectorizer:
    """
    The underlying logic for vectorizing documents and storing them into local or cloud storage.
    Standalone module that can be used by ../agents/vectorize.py VectorStoreCreator. 
    Meant to provide integrationg to both Azure and Vertex AI cloud vector storage.
    """
    def __init__(self, llm_picker: LLMPicker) -> None:
        """
        Initializes the Vectorizer object.

        Args:
            llm_picker (LLMPicker): An object of LLMPicker where the user preferred type of llm, embed model, and chunk size can be customized.
        """
        self._embed_model = llm_picker.embed_model
        self._llm = llm_picker.llm
        self._transformations = self.document_transformations()
        self.service_context = llm_picker.get_default_service_context()
        self.storage_context = None
        self.vector_index = None

    def document_transformations(self) -> t.List:
        transformations = [
            SemanticSplitterNodeParser(embed_model=self._embed_model),
            SummaryExtractor(llm=self._llm, summaries=["self"]),
            self._embed_model,
        ]
        self._transformations = transformations
        return transformations

    def _transform_documents(self, documents: t.List[Document]) -> t.List[Document]:
        pipeline = IngestionPipeline(transformations=self._transformations)
        nodes = pipeline.run(documents=documents, show_progress=True)

        return nodes

    def _build_vector_index(
        self,
        documents: t.List[Document],
    ) -> None:
        with Timer() as vector_index_timer:
            self.vector_index = VectorStoreIndex.from_documents(
                documents=documents,
                transformations=self._transformations,
                service_context=self.service_context,
                show_progress=True,
                storage_context=getattr(self, "storage_context", None),
            )
        logger.info(
            f"Vectorized all the nodes in {vector_index_timer.exec_time / 60} mins"
        )

    def _merge_new_docs(self, documents: t.List[Document]) -> None:
        if not hasattr(self, "vector_index"):
            raise AttributeError(
                "No pre-existing vector index available, build one first"
            )

        with Timer() as vector_index_update_timer:
            self.vector_index.insert_nodes(self._transform_documents(documents))
        logger.info(
            f"Updated vector index sucessfully in {vector_index_update_timer.exec_time / 60} mins."
        )

    def build_and_store_local(
        self,
        documents: t.List[Document],
        index_path: str | None = None,
        merge: bool = False,
    ) -> VectorStoreIndex:
        if index_path is None:
            raise ValueError("Please provide a path to persist the index")
        if merge:
            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"Vector index not found at {index_path}. Does not exist."
                )
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            self.vector_index = load_index_from_storage(
                storage_context, service_context=self.service_context
            )
            self._merge_new_docs(documents)
        else:
            self._build_vector_index(documents)

        with Timer() as vector_index_write_timer:
            self.vector_index.storage_context.persist(persist_dir=index_path)
        logger.info(
            f"Flushed all the indices to disk in {vector_index_write_timer.exec_time / 60} mins"
        )
        return self.vector_index

    def build_and_store_azure(
        self,
        documents: t.List[Document],
        filterable_metadata_fields: t.Optional[list[str]] = None,
        storage_client: CogSearchClient | None = None,
        index_name: str | None = None,
        merge: bool = False,
    ) -> VectorStoreIndex:
        storage_client = storage_client or CogSearchClient
        cog_client = storage_client.from_defaults(
            index_name=index_name, filterable_metadata_fields=filterable_metadata_fields
        )

        if merge:
            self.vector_index = VectorStoreIndex.from_vector_store(
                cog_client.vector_store, service_context=self.service_context
            )
            self._merge_new_docs(documents)
        else:
            # TODO: how to make sure that the index name doesn't already exist?
            self.storage_context = StorageContext.from_defaults(
                vector_store=cog_client.vector_store
            )
            self._build_vector_index(documents)
        return self.vector_index
    def build_and_store_vertex(
        self,
        documents: t.List[Document],
        documents_version: t.Optional[str] = None,
        storage_client: VertexAIClient | None = None,
        index_name: str | None = None,
        vertex_index_endpoint: str | None = None,
        merge: bool = False
    ) -> None:
        # TODO: test vertext AI integration
        if vertex_index_endpoint is None or index_name is None:
            raise ValueError("Please provide an new/ existing index name and index endpoint for Vertex AI.")

        storage_client = storage_client or VertexAIClient
        cog_client = storage_client.from_defaults(
            index_name=index_name,
            index_endpoint=vertex_index_endpoint
        )

        if merge:
            self.vector_index = VectorStoreIndex.from_vector_store(cog_client.vector_store, service_context=self.service_context)
            self._merge_new_docs(documents)
        else:
            self.storage_context = StorageContext.from_defaults(vector_store=cog_client.vector_store)
            self._build_vector_index(documents)
