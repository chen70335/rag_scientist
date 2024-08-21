from os import getenv
from typing import Optional

from llama_index.core import ServiceContext
from llama_index.core.base.llms.base import BaseLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ds_rag_framework.llms.models import AzureOpenAIMixin, OpenAIMixin, VertexAIMixin, HuggingFaceMixin

class LLMPicker(AzureOpenAIMixin, OpenAIMixin, VertexAIMixin, HuggingFaceMixin):
    def __init__(
        self, 
        llm_provider: str,
        llm_model_version: Optional[str] = None,
        chunk_size: int = 1024,
        embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        temperature: float = 0,
        max_retries: int = 30,
        stream: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the LLM model based on the llm_provider parameters.

        Args:
            llm_provider (str): The LLM provider to use. Supported providers are:
                - 'azureopenai': Azure OpenAI provider.
                - 'openai': OpenAI provider.
                - 'palm': PALM provider.
                - 'vertex': Google Vertex AI provider.
                - 'huggingface': HuggingFace provider.
            llm_model_engine_env (Optional[str]): The name of the .env variable for the LLM model engine.
                Applicable for: azureopenai and vertex ai providers.
            llm_model_api_key_env (Optional[str]): The name of the .env variable for the LLM model API key.
                Applicable for: azureopenai and vertex ai providers.
            llm_model_api_version_env (Optional[str]): The name of the .env variable for the LLM model API version.
                Applicable for: azureopenai
            llm_model_endpoint_env (Optional[str]): The name of the .env variable for the LLM model endpoint.
                Applicable for: azureopenai and vertext ai (project id).
            chunk_size (int): The size of each chunk of text to process. Defaults to 1024.
            embed_model_name (str): The name of the embedding model to use. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            temperature (float): The temperature parameter for generating text. Defaults to 0.
            max_retries (int): The maximum number of retries for API requests. Defaults to 30.
            stream (bool): Whether to stream the response or not. Defaults to True.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the specified LLM provider is not supported or if required API keys are missing.

        Notes:
            - For 'azureopenai' provider, the following environment variables are required:
                - llm_model_engine_env: The .env variable name for the LLM model engine.
                - llm_model_api_key_env: The .env variable name for the LLM model API key.
                - llm_model_api_version_env: The .env variable name for the LLM model API version.
                - llm_model_endpoint_env: The .env variable name for the LLM model endpoint.

            - For 'vertex' provider, the following environment variables are required:
                - llm_model_api_key_env: The .env variable name for Vertex model api key.
                - llm_model_engine_env: The .env variable name for Vertex model base engine.
                - llm_model_endpoint_env: The .env variable name for vertex project id.
            - For 'openai' and 'huggingface', please make sure to provide required .env variables in .env.
        """
        self._chunk_size = chunk_size
        self._embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self._llm_model_version = llm_model_version
        self._llm = None
        llm_provider = llm_provider.lower()
        self._llm_provider = llm_provider
        curr_supported_llms = ['azureopenai', 'openai', 'vertex', 'huggingface']
        if llm_provider not in curr_supported_llms:
            raise ValueError(
                
                f"LLM provider {llm_provider} not supported. Supported LLM providers are: {curr_supported_llms}"
            
            )
        # TODO: test implementation with actual Google Vertex AI integration
        # https://docs.llamaindex.ai/en/stable/examples/agent/agentic_rag_using_vertex_ai/
        try: 
            self._llm = getattr(self, f"_configure_{llm_provider}")(
                model_version=llm_model_version,
                temperature=temperature,
                max_retries=max_retries,
                stream=stream,
                **kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error creating {llm_provider} LLM: {e}")


    def get_default_service_context(self) -> ServiceContext:
        service_context = ServiceContext.from_defaults(
            llm=self._llm, chunk_size=1024, embed_model=self._embed_model
        )

        return service_context

    @property
    def llm(self) -> BaseLLM:
        return self._llm

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        return self._embed_model
    
