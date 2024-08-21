"""
Goal: determine best retrieval method for performance (specifically for AzureAI): default (vector dense), hybrid, semantic_hybrid
"""
import os
from os import getenv

from typing import List, Dict, Any, Optional

import asyncio

import pandas as pd
from llama_index.core import ServiceContext, VectorStoreIndex, Document
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.azure_openai import AzureOpenAI


from ds_rag_framework.agents.chat import ChatAgent
from ds_rag_framework.llms.prompt_engineering import PromptDesigner
from ds_rag_framework.autorag.eval_utils import eval_and_generate_metrics

class RetrievalEvaluator:
    """
    Class for evaluating retrieval methods in Azure AI Search Vector Store.
    This class provides functionality to evaluate the performance of different retrieval methods in Azure AI Search Vector Store. It supports three retrieval methods: default, hybrid, and semantic hybrid. The evaluation is based on the metrics of faithfulness and accuracy.
    Attributes:
        llm (BaseLLM): The language model used for evaluation.
        ground_truth_dataset (str): The path to the ground truth dataset in CSV format.
        index_name (str): The name of the existing index on Azure cognitive Search to be used for retrieval.
    Methods:
        create_azure_agent_with_retrieval_method(retrieval_method: str) -> None:
        run_experiment(retrieval_methods: List[str]) -> dict[str, Dict[str, float]]:
        Initialize the RetrievalEvaluator.
        Args:
            llm (BaseLLM): The language model used for evaluation.
            ground_truth_dataset (str): The path to the ground truth dataset in CSV format.
            index_name (str): The name of the existing index on Azure cognitive Search to be used for retrieval.
        Raises:
            ValueError: If the ground truth dataset does not exist or is not in the required CSV format.
        ...
        ...
        ...
    """
    def __init__(
        self,
        llm: BaseLLM, 
        ground_truth_dataset: str,
        index_name: str
        # splitter: add parameter 
    ) -> None:
        _, file_extension = os.path.splitext(ground_truth_dataset)
        if not os.path.exists(ground_truth_dataset) or not file_extension.lower() == '.csv':
            raise ValueError(f"Ground truth dataset {ground_truth_dataset} does not exist or file format is not in required csv format.")
        gt_df = pd.read_csv(ground_truth_dataset)
        self._eval_questions = gt_df['question'].tolist()
        self._eval_ground_truth = gt_df['ground_truth'].tolist()

        self._llm = llm
        self._index_name = index_name

    async def create_azure_agent_with_retrieval_method(
        self,
        retrieval_method: str
    ) -> None:
        """
        Create an Azure agent with the specified retrieval method.
        
        Parameters:
            index_name (str): The name of an existing index on Azure cognitive Searchto be used for retrieval.
            retrieval_method (str): The retrieval method to be used by the agent.
            
        Returns:
            None
        """
        prompt = PromptDesigner.get_prompt(
            use_template=True,
            document_type='html'
        )
        agent = ChatAgent.from_cog_search(
            system_prompt=prompt,
            index_path=self._index_name,
            vector_store_query_mode=retrieval_method
        )

        chat_engine = agent.chat_engine

        eval_results = await eval_and_generate_metrics(
            query_engine=chat_engine,
            llm=self._llm,
            eval_questions=self._eval_questions,
            eval_ground_truth=self._eval_ground_truth,
        )

        print(retrieval_method, eval_results)

        return eval_results

    async def run_experiment(
        self,
        retrieval_methods: List[str]
    ) -> dict[str, Dict[str, float]]:
        """
        Run an experiment to evaluate the performance of different retrieval methods.
        
        Parameters:
            retrieval_methods (List[str]): A list of retrieval methods to be evaluated.
            
        Returns:
            dict[str, Dict[str, float]]: A dictionary containing the evaluation results for each retrieval method.
        """
        results = {}
        for method in retrieval_methods:
            results[method] = await self.create_azure_agent_with_retrieval_method(
                retrieval_method=method
            )

        return results

if __name__ == '__main__':
    async def main():
        ground_truth_dataset = 'src/ds_rag_framework/autorag/test_data_payor24.1.csv'
        llm = AzureOpenAI(
            engine=getenv('OPENAI_MODEL_ENGINE4'),
            api_key=getenv('OPENAI_API_KEY4'),
            api_version=getenv('OPENAI_API_VERSION4'),
            azure_endpoint=getenv('OPENAI_API_BASE4')
        )
        print(f"llm: {llm}")
        evaluator = RetrievalEvaluator(llm, ground_truth_dataset, index_name="documentation-html")
        retrieval_methods = ['default', 'hybrid', 'semantic_hybrid']
        results = await evaluator.run_experiment(retrieval_methods)
        print("results: ", results)
    asyncio.run(main())

    # results
    # default {'avg_time': 2.9405941645304363, 'avg_faithfulness': 1.0, 'avg_relevancy': 0.9333333333333333, 'avg_correctness': 4.533333333333333}
    # hybrid {'avg_time': 3.020539649327596, 'avg_faithfulness': 1.0, 'avg_relevancy': 1.0, 'avg_correctness': 4.433333333333334}


        
        


        
