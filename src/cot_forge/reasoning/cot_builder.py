"""
Implementation of the CoTBuilder class, which is the main abstraction which
is used to handle the control flow to create CoT using sampling and search.
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

# TODO: Add write to file methods and, logging, and checkpoints
# TODO: Add __str__ / __repr__ methods to all classes (scorer, verifier, cot_builder, search)

class CoTBuilder:
    """
    This class contains states and logic to use LLMs to construct 
    chains of thought (CoT) that, through reasoning, connect premises
    to ground truth answers.
    """
    
    def __init__(self,
                 reasoning_llm: LLMProvider,
                 search: SearchAlgorithm,
                 verifier: BaseVerifier,
                 scorer: BaseScorer = None,
                 strategy_reg: StrategyRegistry = default_strategy_registry,
                 search_llm_kwargs: dict[str, Any] = None,
                 ):
        """
        Initialize the CoTBuilder with an LLM provider, search algorithm,
        and strategy registry.
        Args:
            reasoning_llm: LLM provider to use for generating cot responses
            search: Search algorithm to use for constructing CoT
            verifier: An instance of a verifier to check if the answer is correct
            scorer: An instance of a scorer to evaluate different paths
            strategy_reg: Registry of strategies to sample from
            search_llm_kwargs: Additional kwargs for LLM provider used in search
        """
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
        Execute the search algorithm to build a CoT.
        
        Args:
            question: The question to be answered
            ground_truth_answer: The true answer to the question
            llm_kwargs: Additional kwargs for LLM provider
            **kwargs: Additional kwargs for search algorithm
        
        Returns:
            A SearchResult containing the chain of thought and metadata
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
        Execute the search algorithm to build a CoT for a batch of questions.
        
        Args:
            questions: List of questions to be answered
            ground_truth_answers: List of true answers to the questions
            llm_kwargs: Additional kwargs for LLM provider
            multi_thread: Whether to run the search in multi-thread mode
            progress_bar: Whether to show a progress bar
            max_workers: Number of workers for multi-thread execution, required if multi_thread is True
            **kwargs: Additional kwargs for search algorithm
        
        Returns:
            A list of SearchResult containing the chain of thought and metadata
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