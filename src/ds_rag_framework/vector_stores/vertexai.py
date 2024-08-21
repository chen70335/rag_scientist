from logging import getLogger
from os import getenv
from typing import Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import (MatchingEngineIndex,
                                     MatchingEngineIndexEndpoint)
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

logger = getLogger(__name__)


aiplatform.init(project=getenv("GOOGLE_PROJECT_ID"), location=getenv("GOOGLE_REGION"))


class VertexAIClient:
    # TODO: test vertext AI integration
    """
    A class to initiate Vertex AI client for vector search using llama index
    Stand-alone module that can be used by ../agents/vectorize.py VectorStoreCreator or
    ../vectorizer/vectorizer.py DocVectorizer. Meant to provide integration to Vertex AI cloud vector storage.
    """
    @classmethod
    def from_defaults(
        cls,
        index_name: Optional[str] = None,
        index_endpoint: Optional[str] = None,
        embedding_dimensionality: int = 768,
    ) -> "VertexAIVectorStore":
        index_name = (
            index_name
            if index_name is not None
            else getenv("GOOGLE_VERTEX_VS_INDEX_NAME")
        )
        index_endpoint = (
            index_endpoint
            if index_endpoint is not None
            else getenv("GOOGLE_VERTEX_VS_INDEX_ENDPOINT")
        )

        if index_name is None or index_endpoint is None:
            raise ValueError(
                "Please provide an index name and index endpoint for Vertex AI as inputs or .env variables."
            )

        vs_index = cls._create_or_use_existing_index(
            index_name, embedding_dimensionality
        )
        vs_endpoint = cls._create_or_use_existing_endpoint(index_endpoint)
        cls._deploy_index_to_endpoint(vs_index, index_name, vs_endpoint)

        return VertexAIVectorStore(
            project_id=getenv("GOOGLE_PROJECT_ID"),
            region=getenv("GOOGLE_REGION"),
            index_id=vs_index.resource_name,
            endpoint_id=vs_endpoint.resource_name,
            gcs_bucket_name=getenv("GOOGLE_VS_BUCKET_NAME"),
        )

    @staticmethod
    def _create_or_use_existing_index(
        index_name: str, embedding_dimensionality: int = 384
    ) -> "MatchingEngineIndex":
        # check if index exists
        index_names = [
            index.resource_name
            for index in aiplatform.MatchingEngineIndex.list(
                filter=f"display_name={index_name}"
            )
        ]

        if len(index_names) == 0:
            logger.info(f"Creating Vector Search index {index_name} ...")
            vs_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=index_name,
                dimensions=embedding_dimensionality,
                distance_measure_type="DOT_PRODUCT_DISTANCE",
                shard_size="SHARD_SIZE_SMALL",
                index_update_method="STREAM_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE
            )
            logger.info(
                f"Vector Search index {vs_index.display_name} created with resource name {vs_index.resource_name}"
            )
        else:
            vs_index = aiplatform.MatchingEngineIndex(index_name=index_names[0])
            logger.info(
                f"Vector Search index {vs_index.display_name} exists with resource name {vs_index.resource_name}"
            )
        return vs_index

    @staticmethod
    def _create_or_use_existing_endpoint(
        index_endpoint: str,
    ) -> "MatchingEngineIndexEndpoint":
        endpoint_names = [
            endpoint.resource_name
            for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
                filter=f"display_name={index_endpoint}"
            )
        ]

        if len(endpoint_names) == 0:
            logger.info(f"Creating Vector Search index endpoint {index_endpoint} ...")
            vs_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=index_endpoint, public_endpoint_enabled=True
            )
            logger.info(
                f"Vector Search index endpoint {vs_endpoint.display_name} created with resource name {vs_endpoint.resource_name}"
            )
        else:
            vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_names[0]
            )
            logger.info(
                f"Vector Search index endpoint {vs_endpoint.display_name} exists with resource name {vs_endpoint.resource_name}"
            )
        return vs_endpoint

    @staticmethod
    def _deploy_index_to_endpoint(
        vs_index: "MatchingEngineIndex",
        index_name: str,
        vs_endpoint: "MatchingEngineIndexEndpoint",
    ) -> None:
        # check if endpoint exists
        index_endpoints = [
            (deployed_index.index_endpoint, deployed_index.deployed_index_id)
            for deployed_index in vs_index.deployed_indexes
        ]

        if len(index_endpoints) == 0:
            logger.info(
                f"Deploying Vector Search index {vs_index.display_name} at endpoint {vs_endpoint.display_name} ..."
            )
            vs_deployed_index = vs_endpoint.deploy_index(
                index=vs_index,
                deployed_index_id=index_name,
                display_name=index_name,
                machine_type="e2-standard-16",
                min_replica_count=1,
                max_replica_count=1,
            )
            logger.info(
                f"Vector Search index {vs_index.display_name} is deployed at endpoint {vs_deployed_index.display_name}"
            )
        else:
            vs_deployed_index = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=index_endpoints[0][0]
            )
            logger.info(
                f"Vector Search index {vs_index.display_name} is already deployed at endpoint {vs_deployed_index.display_name}"
            )

