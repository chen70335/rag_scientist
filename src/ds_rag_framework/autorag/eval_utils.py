import logging
import time
from typing import Optional

from llama_index.core import ServiceContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.simple import BaseChatEngine
from llama_index.core.evaluation import (BatchEvalRunner, CorrectnessEvaluator,
                                         FaithfulnessEvaluator,
                                         RelevancyEvaluator)
from llama_index.core.llms.llm import BaseLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def eval_and_generate_metrics(
    query_engine: BaseQueryEngine | BaseChatEngine,
    llm: BaseLLM,
    eval_questions: list[str],
    eval_ground_truth: list[str],
    embed_model_name: Optional[str] = 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: int = 1024,
    eval_faithfulness: bool = True,
    eval_relevancy: bool = True,
    eval_correctness: bool = True
) -> dict[str, float]:
    """
    Evaluates the performance of the RAG (Retrieval-Augmented Generation) model by generating metrics based on the given llm query engine and a question answer dataset.
    Parameters:
        query_engine (BaseQueryEngine | BaseChatEngine): The query engine used for retrieving responses.
        llm (BaseLLM): The language model used for generating responses.
        eval_questions (list[str]): A list of evaluation questions.
        eval_ground_truth (list[str]): A list of ground truth answers corresponding to the evaluation questions.
        embed_model_name (Optional[str], optional): The name of the embedding model used for encoding the questions and answers. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        chunk_size (int, optional): The size of each chunk for chunk splitting. Defaults to 1024.
        eval_faithfulness (bool, optional): Whether to evaluate faithfulness. Defaults to True.
        eval_relevancy (bool, optional): Whether to evaluate relevancy. Defaults to True.
        eval_correctness (bool, optional): Whether to evaluate correctness. Defaults to True.
    Returns:
        dict[str, float]: A dictionary containing the average response time, average faithfulness score, average relevancy score, and average correctness score.
    Raises:
        None
    """
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    service_ctx4 = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size,
                                                embed_model=embed_model)
    eval_runner = BatchEvalRunner(
        {
            'faithfulness': FaithfulnessEvaluator(service_context=service_ctx4) if eval_faithfulness else None,
            'relevancy': RelevancyEvaluator(service_context=service_ctx4) if eval_relevancy else None,
            'correctness': CorrectnessEvaluator(service_context=service_ctx4) if eval_correctness else None
        },
        show_progress=True
    )


    num_questions = len(eval_questions)
    total_response_time = 0.0
    responses = []
    for question in eval_questions:
        start_time = time.time()
        res = query_engine.query(question)
        elapsed_time = time.time() - start_time
        total_response_time += elapsed_time
        responses.append(res)

    eval_results = await eval_runner.aevaluate_responses(
        queries=eval_questions,
        responses=responses,
        reference=eval_ground_truth
    )

    avg_time = total_response_time / num_questions
    avg_faithfulness = _help_generate_metrics_average_score('faithfulness', eval_results)
    avg_relevancy = _help_generate_metrics_average_score('relevancy', eval_results)
    avg_correctness = _help_generate_metrics_average_score('correctness', eval_results)

    logger.info(
        f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, "
        f"Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: "
        f"{avg_relevancy:.2f}, Average Correctness: {avg_correctness:.2f}"
    )

    return {
        'avg_time': avg_time,
        'avg_faithfulness': avg_faithfulness,
        'avg_relevancy': avg_relevancy,
        'avg_correctness': avg_correctness
    }

def _help_generate_metrics_average_score(
    key: str,
    eval_results: dict[str, list]
):
    results = eval_results[key]
    total_score = 0
    for result in results:
        total_score += result.score
    return total_score / len(results)


