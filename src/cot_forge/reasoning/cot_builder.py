"""
This module defines the CoTBuilder class, the central component for constructing Chains of Thought (CoTs) 
using Language Model (LLM) sampling and search algorithms.

The CoTBuilder orchestrates the process of generating reasoning steps to connect a given 
question to its ground truth answer. It leverages configurable search algorithms, verifiers, scorers, 
and strategy registries to explore and evaluate potential CoTs.

Key components:
    - CoTBuilder: The main class for building CoTs. It manages the search process, 
        utilizing an LLM, search algorithm, verifier, and scorer.
    - SearchAlgorithm: An interface for search algorithms that explore the space of possible CoTs.
    - BaseVerifier: An interface for verifying the correctness of a CoT's answer.
    - BaseScorer: An interface for scoring different CoT paths during the search.
    - StrategyRegistry: A registry of strategies used to sample different reasoning steps.

Usage:
    1. Instantiate a CoTBuilder with a reasoning LLM, search algorithm, verifier, 
        and optional scorer and strategy registry.
    2. Call the `build` method with a question and its ground truth answer to generate a CoT.
    3. Alternatively, use the `build_batch` method to process multiple questions 
        in single-threaded or multi-threaded mode.

Example:
    ```python

    # Initialize components (replace with your actual implementations)
    search_llm = GeminiProvider(...)
    search_algorithm = NaiveLinearSearch(...)
    verifier = LLMJudgeVerifier()

    # Instantiate CoTBuilder
    cot_builder = CoTBuilder(
        search_llm=search_llm,
        search=search_algorithm,
        verifier=verifier
    )

    # Build a CoT for a single question
    question = "What is the capital of France?"
    ground_truth_answer = "Paris"
    search_result = cot_builder.build_cot(question, ground_truth_answer)

    # Access the generated CoT
    cot = search_result.get_successful_terminal_nodes()[0].get_full_cot()
    print(cot)
    ```
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from cot_forge.llm import LLMProvider
from cot_forge.persistence import PersistenceManager
from cot_forge.post_processing import ReasoningProcessor
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.types import SearchResult
from cot_forge.reasoning.verifiers import BaseVerifier

from .search.search_algorithm import SearchAlgorithm
from .strategies import StrategyRegistry, default_strategy_registry

# TODO: Consider what to do wrt overwriting/duplicate result data

logger = logging.getLogger(__name__)

class CoTBuilder:
    """
    Constructs verifiable chains of thought using language models.

    A chain of thought (CoT) is a sequence of reasoning steps that connects a question
    to its answer through logical deduction. This class manages the process of:
    1. Generating candidate reasoning steps using LLMs
    2. Searching through possible reasoning paths
    3. Verifying correctness of conclusions

    Attributes:
        search_llm (LLMProvider): Language model for generating reasoning steps in search
        search (SearchAlgorithm): Algorithm for exploring reasoning paths
        verifier (BaseVerifier): Validates reasoning conclusions
        scorer (BaseScorer): Evaluates path quality to prioritize exploration
        strategy_registry (StrategyRegistry): Available reasoning strategies
        search_llm_kwargs (dict): Additional LLM configuration
        post_processing_llm (LLMProvider): Language model for post-processing reasoning steps
        post_processing_llm_kwargs (dict): Additional LLM configuration for post-processing
        persistence (PersistenceManager): Manages data storage and retrieval
        post_processor (ReasoningProcessor): Post-processes reasoning steps into natural language

    Example:
        ```python
        builder = CoTBuilder(
            search_llm=llm,
            search=search_algo,
            verifier=verifier
        )
        result = builder.build_cot("Why is the sky blue?", "Rayleigh scattering")
        ```
    """
    
    def __init__(
        self,
        search_llm: LLMProvider,
        search: SearchAlgorithm,
        verifier: BaseVerifier,
        post_processing_llm: LLMProvider,
        dataset_name: str,
        base_dir: str = "data",
        scorer: BaseScorer = None,
        strategy_registry: StrategyRegistry = default_strategy_registry,
        search_llm_kwargs: dict[str, Any] = None,
        post_processing_llm_kwargs: dict[str, Any] = None,
    ):
        self.search_llm = search_llm
        self.search_llm_kwargs = search_llm_kwargs or {}
        self.post_processing_llm = post_processing_llm
        self.post_processing_llm_kwargs = post_processing_llm_kwargs or {}
        self.strategy_registry = strategy_registry
        self.search = search
        self.verifier = verifier
        self.scorer = scorer
        
        # Initialize persistence
        self.persistence = PersistenceManager(
            dataset_name=dataset_name,
            search_name=search.name,
            base_dir=base_dir,
        )
        # Save the configuration
        self.persistence.save_config(self)

        # Initialize post-processor
        self.post_processor = ReasoningProcessor(
            llm_provider=post_processing_llm,
            dataset_name=dataset_name,
            search_name=search.name,
            llm_kwargs=post_processing_llm_kwargs or {},
            base_dir=base_dir,
            output_file="processed_reasoning.jsonl",
            thinking_tag="thinking"
        )
        
    def build_cot(
        self,
        question: str,
        ground_truth_answer: str,
        llm_kwargs: dict[str, Any] = None,
        **kwargs
        ) -> SearchResult | None:
        """
        Use a search algorithm to build chain of thought search result for a single question.

        Uses the configured search algorithm to generate and validate a reasoning path
        that connects the question to its known answer.

        Args:
            question (str): The question requiring explanation
            ground_truth_answer (str): Known correct answer
            llm_kwargs (dict[str, Any], optional): Additional LLM parameters
            **kwargs: Additional parameters for search algorithm

        Returns:
            SearchResult: Contains:
                - terminal_nodes: List of terminal nodes reached in search
                - succes: Boolean indicating if a valid path was found
                - metadata: Search statistics and configuration
            Or None if the question is skipped because it was already processed.

        Example:
            ```python
            result = builder.build_cot(
                question="How many continents are there?",
                ground_truth_answer="7",
                temperature=0.7
            )
            print(result.get_successful_terminal_nodes[0].get_full_cot())
            ```
        """
        # Check if the question has already been processed
        if (self.persistence and 
            self.persistence.should_skip(question, ground_truth_answer)
        ):
            logger.info(
                f"Skipping already processed question: "
                f"{self.persistence.generate_question_id(question, ground_truth_answer)}"
            )

            return None
        
        llm_kwargs = llm_kwargs or {}
        result = self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=self.verifier,
            scorer=self.scorer,
            search_llm=self.search_llm,
            llm_kwargs=llm_kwargs,
            strategy_registry=self.strategy_registry,
            **kwargs
        )
        
        return result
    
    def process(
        self,
        question: str,
        ground_truth_answer: str,
        only_successful: bool = True,
        llm_kwargs: dict[str, Any] | None = None,
        post_processing_llm_kwargs: dict[str, Any] | None = None,
        **kwargs
    ) -> tuple[SearchResult, dict]:
        """
        Process a single question and ground truth answer through full pipeline.
        This includes building the CoT and post-processing the results into natural language.
        Requires a post-processor to not be none.
        This method is a wrapper around `build` to provide a consistent interface.
        
        Args:
            question (str): The question to process
            ground_truth_answer (str): The known correct answer
            only_successful (bool, optional): If True, only process successful results
            llm_kwargs (dict[str, Any] | None, optional): Additional LLM parameters
            post_processing_llm_kwargs (dict[str, Any] | None, optional): Additional LLM parameters for post-processing
            **kwargs: Additional parameters for search algorithm
        Returns:    
            tuple[SearchResult, dict]: A tuple containing:
                - search_result: The result of the search algorithm
                - reasoning: The processed reasoning steps in natural language
        """
        
        if self.post_processor is None:
            raise ValueError("Post-processor is required for processing batch.")
        
        # Build the CoT
        search_result = self.build_cot(
            question=question,
            ground_truth_answer=ground_truth_answer,
            llm_kwargs=llm_kwargs,
            **kwargs
        )
        
        id = self.persistence.generate_question_id(
            question=question,
            ground_truth=ground_truth_answer
        )
        
        # Process the results using the post-processor
        reasoning = self.post_processor.process_result(
            search_result=search_result,
            id=id,
            only_successful=only_successful,
            llm_kwargs=post_processing_llm_kwargs,
            **kwargs
        )
        
        # Save the search result and reasoning
        self.persistence.save_result(
            result=search_result,
            reasoning=reasoning,
            question=question,
            ground_truth=ground_truth_answer,
        )
        
        return search_result, reasoning
        
    def process_batch(
        self,
        questions: list[str],
        ground_truth_answers: list[str],
        llm_kwargs: dict[str, Any] | None = None,
        multi_thread: bool = False,
        progress_bar: bool = True,
        max_workers: int | None = 4,
        load_processed: bool = False,
        overwrite: bool = False,
        limit: int | None = None,
        only_successful: bool = True,
        **kwargs
    ) -> list[tuple[SearchResult, dict]]:
        """
        Process multiple questions in batch mode.

        Supports both single-threaded and multi-threaded execution with progress tracking.

        Args:
            questions (list[str]): List of questions to process
            ground_truth_answers (list[str]): Corresponding correct answers
            llm_kwargs (dict[str, Any] | None, optional): Additional LLM parameters
            multi_thread (bool, optional): Enable parallel processing
            progress_bar (bool, optional): Show progress indicator
            max_workers (int | None, optional): Thread pool size for parallel processing
            overwrite (bool, optional): If True, deletes all existing files before processing
            load_processed (bool, optional): If True load already processed results from disk
            limit (int | None, optional): Limit the number of questions to process
            only_successful (bool, optional): If True, only process successful results into natural language
            **kwargs: Additional search algorithm parameters

        Returns:
            A list of tuples containing:
                - search_result: The result of the search algorithm
                - reasoning: The processed reasoning steps in natural language

        Performance:
            - Single-threaded: Processing is sequential but memory-efficient
            - Multi-threaded: Faster for large batches, higher memory usage

        Example:
            ```python
            questions = ["What is 2+2?", "What is 3+3?"]
            answers = ["4", "6"]
            results = builder.process_batch(
                questions=questions,
                ground_truth_answers=answers,
                multi_thread=True,
                max_workers=4
            )
            ```
        """
        
        if len(questions) != len(ground_truth_answers):
            raise ValueError("Questions and ground truth answers must have the same length.")
        if multi_thread and max_workers is None:
            raise ValueError("max_workers must be specified when multi_thread is True.")
        
        # If overwrite is set to True, delete all existing files
        if overwrite:
            self.persistence.reset_all_files()
            self.persistence.save_config(self)

        # Set up persistence for batch processing
        self.persistence.setup_batch_run()

        results = []
        # Load processed results if requested
        if load_processed:
            result_dicts = self.persistence.load_search_results()
            reasoning = self.persistence.load_reasoning()
            
            # Assume the strategy registry for saved results is same as current instance
            if result_dicts:
                search_results = [
                    SearchResult.deserialize(item["result"], self.strategy_registry) for item in result_dicts
                ]
                logger.info(f"Loaded {len(search_results)} processed results from disk.")
            else:
                search_results = []
                logger.info("No processed results found on disk.")           
            results = list(zip(search_results, reasoning)) if load_processed else []
        
        # Get q_a pairs to process by removing already processed ones
        qa_pairs = [
            (q, a) for q,a in zip(questions, ground_truth_answers)
            if not self.persistence.should_skip(q, a)
        ]
        
        # Limit the number of questions to process if specified
        if limit is not None:
            qa_pairs = qa_pairs[:limit]
        
        if multi_thread:
            new_results = self._multi_thread_batch_process(
                qa_pairs=qa_pairs,
                progress_bar=progress_bar,
                max_workers=max_workers,
                llm_kwargs=llm_kwargs,
                only_successful=only_successful,
                **kwargs
            )
            return results + new_results
        else:
            new_results = self._single_threaded_batch_process(
                qa_pairs=qa_pairs,
                progress_bar=progress_bar,
                llm_kwargs=llm_kwargs,
                only_successful=only_successful,
                **kwargs
            )
            return results + new_results

    def _single_threaded_batch_process(
        self,
        qa_pairs: list[tuple[str, str]],
        progress_bar: bool,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs
    ) -> list[SearchResult]:
        """Execute the search algorithm to build a CoT for a batch of questions in single-threaded mode."""
        results = []
        if progress_bar:
                qa_pairs = tqdm(
                    qa_pairs,
                    total=len(qa_pairs),
                    desc="Single-threaded processing question and ground truth answer pairs.",
                    unit="pair"
                )
        for q, a in qa_pairs:
            result, reasoning = self.process(
                question=q,
                ground_truth_answer=a,
                llm_kwargs=llm_kwargs,
                **kwargs
            )
            results.append((result, reasoning))
            
        return results
    
    def _multi_thread_batch_process(
        self,
        qa_pairs: list[tuple[str, str]],
        progress_bar: bool,
        max_workers: int,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs
    ) -> list[SearchResult]:
        """Execute the search algorithm to build a CoT for a batch of questions in multi-thread mode."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.process,
                    question=q,
                    ground_truth_answer=a,
                    llm_kwargs=llm_kwargs,
                    **kwargs
                )
                for q, a in qa_pairs
            ]
            
            future_iterator = futures
            if progress_bar:
                future_iterator = tqdm(
                    futures,
                    total=len(futures),
                    desc="Multi-thread processing question and ground truth answer pairs.",
                    unit="pair"
                )
            
            for future in future_iterator:
                search_result, reasoning = future.result()
                results.append((search_result, reasoning))
                
        return results

    def __repr__(self) -> str:
        persistence_info = (
            f"\n\tPersistence: Enabled ({self.persistence.dataset_name})" 
            if self.persistence else "\n\tPersistence: Disabled"
        )
        return (
            f"CoTBuilder with:{persistence_info}\n"
            f"\tLLM: {self.search_llm}\n"
            f"\tSearch Algorithm: {self.search}\n"
            f"\tVerifier: {self.verifier}\n"
            f"\tScorer: {self.scorer}\n"
            f"\tStrategy Registry: {self.strategy_registry}\n"
            f"\tSearch LLM Kwargs: {self.search_llm_kwargs}\n"
            f"\tPost-Processing LLM: {self.post_processing_llm}\n"
            f"\tPost-Processing LLM Kwargs: {self.post_processing_llm_kwargs}"
        )
    
    def __str__(self) -> str:
        return (
            f"CoTBuilder with:\n"
            f"\tLLM: {self.search_llm}\n"
            f"\tSearch Algorithm: {self.search}\n"
            f"\tVerifier: {self.verifier}\n"
            f"\tScorer: {self.scorer}\n"
            f"\tStrategies: {self.strategy_registry.list_strategies()}\n"
            f"\tPost-Processing LLM: {self.post_processing_llm}"
        )