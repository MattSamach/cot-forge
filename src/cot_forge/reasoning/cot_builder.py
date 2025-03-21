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
    reasoning_llm = GeminiLLMProvider(...)
    search_algorithm = NaiveLinearSearch(...)
    verifier = LLMJudgeVerifier()

    # Instantiate CoTBuilder
    cot_builder = CoTBuilder(
        reasoning_llm=reasoning_llm,
        search=search_algorithm,
        verifier=verifier
    )

    # Build a CoT for a single question
    question = "What is the capital of France?"
    ground_truth_answer = "Paris"
    search_result = cot_builder.build(question, ground_truth_answer)

    # Access the generated CoT
    cot = search_result.get_successful_terminal_nodes()[0].get_full_cot()
    print(cot)
    ```
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from cot_forge.llm import LLMProvider
from cot_forge.reasoning.scorers import BaseScorer
from cot_forge.reasoning.verifiers import BaseVerifier

from .search.search_algorithm import SearchAlgorithm, SearchResult
from .strategies import StrategyRegistry, default_strategy_registry

# TODO: Add write to file methods, logging, and checkpoints

class CoTBuilder:
    """
    Constructs verifiable chains of thought using language models.

    A chain of thought (CoT) is a sequence of reasoning steps that connects a question
    to its answer through logical deduction. This class manages the process of:
    1. Generating candidate reasoning steps using LLMs
    2. Searching through possible reasoning paths
    3. Verifying correctness of conclusions

    Attributes:
        reasoning_llm (LLMProvider): Language model for generating reasoning steps
        search (SearchAlgorithm): Algorithm for exploring reasoning paths
        verifier (BaseVerifier): Validates reasoning conclusions
        scorer (BaseScorer): Evaluates path quality to prioritize exploration
        strategy_reg (StrategyRegistry): Available reasoning strategies
        search_llm_kwargs (dict): Additional LLM configuration

    Example:
        ```python
        builder = CoTBuilder(
            reasoning_llm=llm,
            search=search_algo,
            verifier=verifier
        )
        result = builder.build("Why is the sky blue?", "Rayleigh scattering")
        ```
    """
    
    def __init__(self,
                 reasoning_llm: LLMProvider,
                 search: SearchAlgorithm,
                 verifier: BaseVerifier,
                 scorer: BaseScorer = None,
                 strategy_reg: StrategyRegistry = default_strategy_registry,
                 search_llm_kwargs: dict[str, Any] = None,
                 ):
        self.reasoning_llm = reasoning_llm
        self.search_llm_kwargs = search_llm_kwargs or {}
        self.strategy_reg = strategy_reg
        self.search = search
        self.verifier = verifier
        self.scorer = scorer
        
    def build(self,
              question: str,
              ground_truth_answer: str,
              llm_kwargs: dict[str, Any] = None,
              **kwargs) -> SearchResult:
        """
        Construct a chain of thought for a single question.

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

        Example:
            ```python
            result = builder.build(
                question="How many continents are there?",
                ground_truth_answer="7",
                temperature=0.7
            )
            print(result.get_successful_terminal_nodes[0].get_full_cot())
            ```
        """
        llm_kwargs = llm_kwargs or {}
        return self.search(
            question=question,
            ground_truth_answer=ground_truth_answer,
            verifier=self.verifier,
            scorer=self.scorer,
            reasoning_llm=self.reasoning_llm,
            llm_kwargs=llm_kwargs,
            strategy_registry=self.strategy_reg,
            **kwargs
        )
        
    def build_batch(self,
                    questions: list[str],
                    ground_truth_answers: list[str],
                    llm_kwargs: dict[str, Any] | None = None,
                    multi_thread: bool = False,
                    progress_bar: bool = True,
                    max_workers: int | None = 4,
                    **kwargs) -> list[SearchResult]:
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
            **kwargs: Additional search algorithm parameters

        Returns:
            list[SearchResult]: List of search results, one per question

        Performance:
            - Single-threaded: Processing is sequential but memory-efficient
            - Multi-threaded: Faster for large batches, higher memory usage

        Example:
            ```python
            questions = ["What is 2+2?", "What is 3+3?"]
            answers = ["4", "6"]
            results = builder.build_batch(
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

        total_pairs = len(questions)
        qa_iterator = iter(zip(questions, ground_truth_answers))
            
        if multi_thread:
            return self._multi_thread_batch_build(
                qa_iterator=qa_iterator,
                llm=self.reasoning_llm,
                progress_bar=progress_bar,
                max_workers=max_workers,
                llm_kwargs=llm_kwargs,
                **kwargs
            )
        else:
            return self._single_threaded_batch_build(
                qa_iterator=qa_iterator,
                llm=self.reasoning_llm,
                progress_bar=progress_bar,
                total_pairs=total_pairs,
                llm_kwargs=llm_kwargs,
                **kwargs
            )    
    
    def _single_threaded_batch_build(self,
                                     qa_iterator: Iterable[tuple[str, str]],
                                     llm: LLMProvider,
                                     progress_bar: bool,
                                     total_pairs: int,
                                     llm_kwargs: dict[str, Any] | None = None,
                                     **kwargs) -> list[SearchResult]:
        """Execute the search algorithm to build a CoT for a batch of questions in single-threaded mode."""
        results = []
        if progress_bar:
                qa_iterator = tqdm(
                    qa_iterator,
                    total=total_pairs,
                    desc="Single-threaded processing question and ground truth answer pairs.",
                    unit="pair"
                )
                
        for q, a in qa_iterator:
            results.append(
                self.build(
                    question=q,
                    ground_truth_answer=a,
                    llm_kwargs=llm_kwargs,
                    **kwargs
                    )
                )
            
        return results
    
    def _multi_thread_batch_build(self,
                                  qa_iterator: Iterable[tuple[str, str]],
                                  llm: LLMProvider,
                                  progress_bar: bool,
                                  max_workers: int,
                                  llm_kwargs: dict[str, Any] | None = None,
                                  **kwargs) -> list[SearchResult]:
            """Execute the search algorithm to build a CoT for a batch of questions in multi-thread mode."""
            
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.build,
                        question=q,
                        ground_truth_answer=a,
                        llm_kwargs=llm_kwargs,
                        **kwargs
                    )
                    for q, a in qa_iterator
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
                    results.append(future.result())
            return results


    def __repr__(self) -> str:
        return (f"CoTBuilder with:\n"
            f"\tLLM: {self.reasoning_llm}\n"
            f"\tSearch Algorithm: {self.search}\n"
            f"\tVerifier: {self.verifier}\n"
            f"\tScorer: {self.scorer}\n"
            f"\tStrategy Registry: {self.strategy_reg}\n"
            f"\tSearch LLM Kwargs: {self.search_llm_kwargs}")
    
    def __str__(self) -> str:
        return (f"CoTBuilder with:\n"
            f"\tLLM: {self.reasoning_llm}\n"
            f"\tSearch Algorithm: {self.search}\n"
            f"\tVerifier: {self.verifier}\n"
            f"\tScorer: {self.scorer}\n"
            f"\tStrategies: {self.strategy_reg.list_strategies()}")