import logging
import os
import sys
import typing as t

from llama_index.core import (ServiceContext, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage

from ds_rag_framework.llms.llm_picker import LLMPicker
from ds_rag_framework.llms.prompt_engineering import PromptDesigner
from ds_rag_framework.utils import (filter_version_at_retrieval,
                                    load_config_json)
from ds_rag_framework.vector_stores.cogsearch import CogSearchClient

logger = logging.getLogger(__name__)


class ChatAgent:
    """
    A class representing a chat agent that interacts with a chat engine.
    Args:
        vector_index (VectorStoreIndex): The vector store index used by the chat engine.
        service_context (ServiceContext): The service context used by the chat engine.
        system_prompt (str | PromptDesigner): The system prompt used by the chat engine.
        similarity_top_k (int, optional): The number of similar responses to retrieve. Defaults to 2.
        version (str | None, optional): The version of the chat agent. Defaults to None.
        **kwargs: Additional keyword arguments.
    Methods:
        query(query: str, chat_history: List[ChatMessage] | None = None, stream: bool = False) -> Any:
            Sends a query to the chat engine and returns the response.
        reset() -> None:
            Resets the chat engine.
        from_local_storage(system_prompt: str | PromptDesigner, index_path: str | None = None,
                           vector_index: VectorStoreIndex | None = None, llm_provider: str = 'azureopenai',
                           llm_model_version: str = '35t', similarity_top_k: int = 3,
                           version: str | None = None, **kwargs) -> ChatAgent:
            Creates a ChatAgent instance from local storage.
        from_cog_search(system_prompt: str | PromptDesigner, index_path: str | None = None,
                        llm_provider: str = 'azureopenai', llm_model_version: str = '35t',
                        similarity_top_k: int = 3, version: str | None = None, **kwargs) -> ChatAgent:
            Creates a ChatAgent instance using CogSearch.
        from_vertex_ai(system_prompt: str | PromptDesigner, index_path: str | None = None,
                       llm_model: str = "vertex", similarity_top_k: int = 3, **kwargs) -> ChatAgent:
            Creates a ChatAgent instance using Vertex AI.
    """
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        service_context: ServiceContext,
        system_prompt: str | PromptDesigner,
        similarity_top_k: int = 2,
        version: str | None = None,
        **kwargs,
    ) -> None:
        filters = filter_version_at_retrieval(version) if version else None

        self.chat_engine = vector_index.as_chat_engine(
            service_context=service_context,
            chat_mode=ChatMode.CONTEXT,
            similarity_top_k=similarity_top_k,
            system_prompt=system_prompt,
            filters=filters,
            **kwargs,
        )

    def query(
        self,
        query: str,
        chat_history: t.List[ChatMessage] | None = None,
        stream: bool = False,
    ) -> t.Any:
        if stream:
            response = self.chat_engine.stream_chat(
                query, chat_history=chat_history if chat_history is not None else None
            )
        else:
            response = self.chat_engine.chat(
                query, chat_history=chat_history if chat_history is not None else None
            )
        if not chat_history:
            self.chat_engine.reset()
        return response

    def reset(self) -> None:
        self.chat_engine.reset()

    @classmethod
    def from_local_storage(
        cls,
        system_prompt: str | PromptDesigner,
        index_path: str | None = None,
        vector_index: VectorStoreIndex | None = None,
        llm_provider: str = 'azureopenai',
        llm_model_version: str = '35t',
        similarity_top_k: int = 3,
        version: str | None = None,
        **kwargs
    ) -> 'ChatAgent':
        llm_picker = LLMPicker(llm_provider=llm_provider, llm_model_version=llm_model_version)
        service_ctx = llm_picker.get_default_service_context()
        if (index_path is None and vector_index is None) or (
            index_path is not None and vector_index is not None
        ):
            raise ValueError(
                "Please provide either an index directory or a vector store, not neither or both."
            )
        elif index_path and not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index directory {index_path} does not exist in {os.getcwd()}"
            )
        if index_path:
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            vector_index = load_index_from_storage(
                storage_context, service_context=service_ctx
            )

        return cls(
            vector_index=vector_index,
            service_context=service_ctx,
            similarity_top_k=similarity_top_k,
            system_prompt=system_prompt,
            version=version,
            **kwargs
        )

    @classmethod
    def from_cog_search(
        cls,
        system_prompt: str | PromptDesigner,
        index_path: str | None = None,
        llm_provider: str = 'azureopenai',
        llm_model_version: str = '35t',
        similarity_top_k: int = 3,
        version: str | None = None,
        **kwargs
    ) -> 'ChatAgent':
        llm_picker = LLMPicker(llm_provider=llm_provider, llm_model_version=llm_model_version)
        service_ctx = llm_picker.get_default_service_context()
        if version:
            filters = ["version"]
        else:
            filters = None
        cog_service = CogSearchClient.from_defaults(index_name=index_path, filterable_metadata_fields=filters)
        storage_context = StorageContext.from_defaults(vector_store=cog_service.vector_store)
        vector_index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, service_context=service_ctx
        )

        return cls(
            vector_index=vector_index,
            service_context=service_ctx,
            similarity_top_k=similarity_top_k,
            system_prompt=system_prompt,
            version=version,
            **kwargs
        )

    @classmethod
    def from_vertex_ai(
        cls,
        system_prompt: str | PromptDesigner,
        index_path: str | None = None,
        llm_model: str = "vertex",
        similarity_top_k: int = 3,
        **kwargs
    ) -> 'ChatAgent':
        # TODO: test vertex integration
        llm_picker = LLMPicker(llm_provider=llm_model)
        service_ctx = llm_picker.get_default_service_context()
        return cls(
            vector_index=index_path,
            service_context=service_ctx,
            similarity_top_k=similarity_top_k,
            system_prompt=system_prompt,
            **kwargs
        )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Please provide the path to the config file. Ex: python src/ds_rag_framework/agents/vectorize.py src/ds_rag_framework/configs/vectorize_config.json")
    config_file_path = sys.argv[1]
    print(config_file_path)
    config = load_config_json(config_file_path)
    param_dict = config['chat_agent']
    param_dict['system_prompt'] = " ".join(config["chat_agent"]["system_prompt"])

    location = param_dict["vector_store_location"]
    if location not in ["local", "cogsearch", "vertex"]:
        raise ValueError(
            "Invalid vector store location. Must be either 'local', 'cogsearch', or 'vertex'."
        )

    if location == "local":
        del param_dict["vector_store_location"]
        agent = ChatAgent.from_local_storage(**param_dict)
    elif location == "cogsearch":
        del param_dict["vector_store_location"]
        agent = ChatAgent.from_cog_search(**param_dict)
    elif location == "vertex":
        del param_dict["vector_store_location"]
        agent = ChatAgent.from_vertex_ai(**param_dict)
    print(f"Chat agent created successfully as {agent}")
    # TODO: determine function of file since UI is now in another file
    # maybe create API endpoint here
