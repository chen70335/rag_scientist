from os import getenv
from typing import Optional

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.openai import OpenAI
from llama_index.llms.vertex import Vertex

from ds_rag_framework.utils import check_env_var

DEFAULT_ENV_ERROR_MESSAGE = (
    ' Please provide an .env variable named {env_var}'
    " for {llm_provider} with model version {llm_model_version}."
    " Check method documentation for description of required"
    " environment variables and naming conventions."
)

class AzureOpenAIMixin:
    """
    Mixin class for configuring the AzureOpenAI model. Not meant for stand-alone use
    but to be used in conjunction with the LLMPicker class.
    """
    def _configure_azureopenai(
        self,
        model_version: Optional[str] = '35t',
        temperature: float = 0,
        max_retries: int = 30,
        stream: bool = True,
        **kwargs
    ) -> AzureOpenAI:
        """
        Configures the AzureOpenAI model with the specified parameters.

        Args:
            model_version (Optional[str], optional): The version of the model to use. Defaults to '35t' (GPT 3.5 Turbo).
                Can also be set to '4' (GPT 4). Defaults to '35t'.
            temperature (float, optional): The temperature parameter for generating responses. Defaults to 0.
            max_retries (int, optional): The maximum number of retries for API calls. Defaults to 30.
            stream (bool, optional): Whether to stream the response or get the entire response at once. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the AzureOpenAI constructor.

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI model configured with the specified parameters.

        """
        if model_version == None:
            model_version = '35t'


        env_variables = {
            'model_engine': f'AZURE_OPENAI_MODEL_ENGINE_{model_version}',
            'api_key': f'AZURE_OPENAI_API_KEY_{model_version}',
            'model_endpoint': f'AZURE_OPENAI_ENDPOINT_{model_version}',
            'api_version': f'AZURE_OPENAI_API_VERSION_{model_version}'
        }

        for env_var in env_variables.keys():
            check_env_var(
                env_variables[env_var],
                DEFAULT_ENV_ERROR_MESSAGE.format(
                    env_var=env_variables[env_var],
                    llm_provider='azureopenai',
                    llm_model_version=model_version
                )
            ) 

        return AzureOpenAI(
            engine=getenv(env_variables['model_engine']),
            api_key=getenv(env_variables['api_key']),
            api_version=getenv(env_variables['api_version']),
            azure_endpoint=getenv(env_variables['model_endpoint']),
            temperature=temperature,
            stream=stream,
            max_retries=max_retries,
            **kwargs
        )

class OpenAIMixin:
    """
    Mixin class for configuring the OpenAI model. Not meant for stand-alone use
    but to be used in conjunction with the LLMPicker class.
    """
    def _configure_openai(
        self,
        model_version: Optional[str] = '35t',
        temperature: float = 0,
        max_retries: int = 30,
        stream: bool = True,
        **kwargs
    ) -> OpenAI:
        if model_version == None:
            model_version = '35t'
        env_variables = {
            'api_key': f"OPENAI_API_KEY_{model_version}",
            'api_base': f"OPENAI_API_BASE_{model_version}",
            'api_version': f"OPENAI_API_VERSION_{model_version}",
        }

        for env_var in env_variables.keys():
            check_env_var(
                env_variables[env_var],
                DEFAULT_ENV_ERROR_MESSAGE.format(
                    env_var=env_variables[env_var],
                    llm_provider='azureopenai',
                    llm_model_version=model_version
                )
            ) 

        return OpenAI(
            api_key=getenv(env_variables['api_key']),
            api_base=getenv(env_variables['api_base']),
            api_version=getenv(env_variables['api_version']),
            temperature=temperature,
            stream=stream,
            max_retries=max_retries,
            **kwargs
        )

class VertexAIMixin:
    """
    Mixin class for configuring the Google Vertex AI model. Not meant for stand-alone use
    but to be used in conjunction with the LLMPicker class.
    """
    # TODO: test implementation w/ Google Cloud API
    def _configure_vertex(
        self,
        model_version: Optional[str] = None,
        temperature: float = 0,
        max_retries: int = 30,
        stream: bool = True,
        **kwargs
    ) -> Vertex:
        env_variables = {
            'project_id': f"GOOGLE_VERTEX_PROJECT_ID_{model_version}",
            'api_key': f"GOOGLE_VERTEX_API_KEY_{model_version}",
            'api_base': f"GOOGLE_VERTEX_API_BASE_{model_version}"
        }

        for env_var in env_variables.keys():
            check_env_var(
                env_variables[env_var],
                DEFAULT_ENV_ERROR_MESSAGE.format(
                    env_var=env_variables[env_var],
                    llm_provider='azureopenai',
                    llm_model_version=model_version
                )
            ) 

        credentials = {
            "project_id": getenv(env_variables['project_id']),
            "api_key": getenv(env_variables['api_key']),
        }
        return Vertex(
            model=getenv(env_variables['api_base']),
            project=credentials['project_id'],
            credentials=credentials,
            temperature=temperature,
            stream=stream,
            max_retries=max_retries,
            **kwargs,
        )

class HuggingFaceMixin:
    """
    Mixin class for configuring the HuggingFace AI model. Not meant for stand-alone use
    but to be used in conjunction with the LLMPicker class.
    """
    # TODO: test huggingface api implementation
    def _configure_huggingface(
        self,
        model_version: Optional[str] = None,
        temperature: float = 0,
        max_retries: int = 30,
        stream: bool = True,
        **kwargs,
    ) -> HuggingFaceInferenceAPI:
        env_variables = {
            'api_base': f"HUGGINGFACE_API_BASE_{model_version}",
            'api_token': f"HF_TOKEN_{model_version}",
        }

        for env_var in env_variables.keys():
            check_env_var(
                env_variables[env_var],
                DEFAULT_ENV_ERROR_MESSAGE.format(
                    env_var=env_variables[env_var],
                    llm_provider='azureopenai',
                    llm_model_version=model_version
                )
            ) 

        return HuggingFaceInferenceAPI(
            model_name=getenv(env_variables['api_base']),
            temperature=temperature,
            stream=stream,
            max_retries=max_retries,
            **kwargs,
        )