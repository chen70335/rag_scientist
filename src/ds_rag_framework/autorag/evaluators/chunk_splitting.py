import os
import time
import typing as t
import logging
from os import getenv

import pandas as pd
from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ds_rag_framework.timer import Timer
from ds_rag_framework.autorag.eval_utils import eval_and_generate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: when should this evaluator be used to be most effective?
class ChunkAndSplitEvaluator:
    """
    Class for evaluating response time and accuracy of a language model
    based on different chunk sizes and splitting techniques. Picks the best chunk size
    and splitting technique for the particular documents.
    Args:
        llm (BaseLLM): The language model to be evaluated.
        ground_truth_dataset (str): The path to the test dataset in CSV format.
        documents (List[Document]): The list of documents to be indexed.
        chunk_sizes (List[int], optional): The list of chunk sizes to be evaluated.
            Defaults to [128, 256, 512, 1024, 2048].
    Methods:
        evaluate_response_time_and_accuracy(chunk_size, splitter, eval_correctness=True,
            eval_faithfulness=True, eval_relevancy=True):
            Evaluates the response time and accuracy of the language model based on
            the given chunk size and splitter.
        evaluate_with_agent():
            Evaluates the response time and accuracy of the language model for each
            chunk size in the list of chunk sizes.
    """
    def __init__(
        self,
        llm: BaseLLM, 
        ground_truth_dataset: str,
        documents: t.List[Document],
        chunk_sizes: t.List[int] = [128, 256, 512, 1024, 2048],
        # splitter: add parameter 
    ) -> None:
        _, file_extension = os.path.splitext(ground_truth_dataset)
        if not os.path.exists(ground_truth_dataset) or not file_extension.lower() == '.csv':
            raise ValueError(f"Ground truth dataset {ground_truth_dataset} does not exist or file format is not in required csv format.")
        # TODO: test data has to be in certain format w/ column names
        gt_df = pd.read_csv(ground_truth_dataset)
        self._eval_questions = gt_df['question'].tolist()
        self._eval_ground_truth = gt_df['ground_truth'].tolist()

        self._llm = llm
        self._documents = documents
        self._chunk_sizes = chunk_sizes


    async def evaluate_response_time_and_accuracy(
        self,
        chunk_size,
        splitter: str,
        eval_correctness: bool = True,
        eval_faithfulness: bool = True,
        eval_relevancy: bool = True
    ):
        embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
        service_context = ServiceContext.from_defaults(
            llm=self._llm,
            chunk_size=chunk_size,
            embed_model=embed_model
        )
        if splitter == "semantic":
            splitter = SemanticSplitterNodeParser(embed_model=embed_model)
        elif splitter == "sentence":
            splitter = SentenceSplitter(chunk_size=chunk_size)


        transformations = [
            splitter,
            embed_model
        ]
        with Timer() as create_index_timer:
            index = VectorStoreIndex.from_documents(
                documents=self._documents,
                transformations=transformations,
                service_context=service_context,
                show_progress=True
            )
            query_engine = index.as_query_engine(service_context=service_context)
        logger.info(f'Created index in {create_index_timer.exec_time / 60} mins')
        
        
        return await eval_and_generate_metrics(
            query_engine=query_engine,
            eval_llm_provider="azure_openai",
            eval_questions=self._eval_questions,
            eval_ground_truth=self._eval_ground_truth,
            eval_llm_model=getenv('OPENAI_MODEL_ENGINE4'),
            chunk_size=chunk_size,
            eval_faithfulness=eval_faithfulness,
            eval_relevancy=eval_relevancy,
            eval_correctness=eval_correctness
        )


    async def evaluate_with_agent(self):
        for chunk_size in self._chunk_sizes:
            await self.evaluate_response_time_and_accuracy(chunk_size, SentenceSplitter)


