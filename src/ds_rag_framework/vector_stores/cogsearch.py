import typing as t
from os import getenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from llama_index.vector_stores.azureaisearch import (
    CognitiveSearchVectorStore, IndexManagement)

load_dotenv(override=True)

CLIENT_MAPPER: dict[str, SearchClient | SearchIndexClient] = {
    "search": SearchClient,
    "index": SearchIndexClient,
}


class CogSearchClient:
    """
    A class to initiate Azure AI client for vector search using llama index
    Stand-alone module that can be used by ../agents/vectorize.py VectorStoreCreator or
    ../vectorizer/vectorizer.py DocVectorizer. Meant to provide integration to Azure AI cloud vector storage.
    """
    @classmethod
    def from_defaults(
        cls,
        index_name: str | None = None,
        filterable_metadata_fields: t.Optional[list[str]] = None,
        embedding_dimensionality: int = 384,
    ) -> "CogSearchClient":
        congnitive_search_creds = AzureKeyCredential(getenv("AZURE_COG_APIKEY"))
        index_client = SearchIndexClient(
            endpoint=getenv("AZURE_COG_URI"),
            credential=congnitive_search_creds,
            index_name=index_name,
        )

        index_name = (
            index_name if index_name is not None else getenv("AZURE_COG_INDEX_NAME")
        )
        return cls(
            index_client=index_client,
            index_name=index_name,
            filterable_metadata_fields=filterable_metadata_fields
            if filterable_metadata_fields
            else None,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field_key="id",
            chunk_field_key="content",
            embedding_field_key="embedding",
            metadata_string_field_key="li_jsonMetadata",
            doc_id_field_key="li_doc_id",
            embedding_dimensionality=embedding_dimensionality,
        )

    def __init__(
        self,
        index_client: SearchIndexClient,
        index_name: str | None,
        filterable_metadata_fields: t.Optional[list[str]] = None,
        **kwargs: t.Any,
    ) -> None:
        self.__vector_store = CognitiveSearchVectorStore(
            search_or_index_client=index_client,
            index_name=index_name,
            filterable_metadata_field_keys=filterable_metadata_fields,
            **kwargs,
        )

    @property
    def vector_store(self) -> CognitiveSearchVectorStore:
        return self.__vector_store
