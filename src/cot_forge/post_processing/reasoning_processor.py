import json
import logging
from pathlib import Path
from typing import Any, Literal

import tqdm

from cot_forge.llm import LLMProvider
from cot_forge.persistence import PersistenceManager
from cot_forge.reasoning.strategies import StrategyRegistry, default_strategy_registry
from cot_forge.reasoning.types import SearchResult
from cot_forge.utils.search_utils import generate_and_parse_json

from .prompts import build_formal_response_prompt, build_natural_language_cot_prompt

logger = logging.getLogger(__name__)

# TODO: Add tests for this module
# TODO: Add write line by line to output file so we can process large files, 
    # ^ also should be something like that in persistence

class ReasoningProcessor:
    """
    Process batches of saved CoT results to apply post-processing transformations.
    
    This class handles loading previously generated CoT results from the persistence system,
    transforming them using a ReasoningProcessor, and saving the transformed results.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        dataset_name: str,
        search_name: str,
        llm_kwargs: dict[str, Any] | None = None,
        base_dir: str = "data",
        output_path: str = None,
        **kwargs
    ):
        """
        Initialize the batch processor.
        
        Args:
            llm_provider: LLM provider for generating natural reasoning and formal responses
            dataset_name: Name of the dataset to process (must match CoTBuilder's dataset_name)
            search_name: Name of the search to process (must match the search algorithm name)
            llm_kwargs: Additional arguments for the LLM provider
            base_dir: Base directory for loading data (should match CoTBuilder's base_dir)
            strategy_reg: Strategy registry for deserializing results
            output_path: Path to save post-processed results:
                (default base_dir/dataset_name/search_name/processed.jsonl)
        """
        self.llm_kwargs = llm_kwargs or {}
        self.llm_provider = llm_provider
        self.dataset_name = dataset_name
        self.search_name = search_name
        self.base_dir = Path(base_dir)
        
        # Create persistence manager to read results (but not write to them)
        self.persistence = PersistenceManager(
            dataset_name=dataset_name,
            search_name=search_name,
            base_dir=base_dir,
            auto_resume=True
        )
        
        # Set output directory
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.persistence.search_dir / "processed_results.jsonl"
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def get_strategy_registry(self) -> StrategyRegistry:
        """
        Get the strategy registry for deserializing results.
        
        Returns:
            StrategyRegistry: The strategy registry
        """
        try:
            return StrategyRegistry.deserialize(self.persistence.load_config()["strategy_registry"])
        except KeyError:
            logger.warning("No strategy registry found in the config. Using default.")
            return default_strategy_registry
        
    def process_batch(
        self,
        only_successful: bool = True,
        limit: int = None,
        progress_bar: bool = True,
        llm_kwargs: dict = None,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> list[dict[str, Any]]:
        """
        Process a batch of saved CoT results.
        
        Args:
            only_successful: If True, only process CoTs that were successful
            limit: Maximum number of results to process
            progress_bar: Whether to show a progress bar
            llm_kwargs: Additional arguments for the LLM
            on_error: Error handling strategy for the processor
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of processed results
        """
        # Load all results
        results_data = self.persistence.load_results()
        
        # Filter for successful results if required
        if only_successful:
            results_data = [r for r in results_data if r["success"]]
            
        if limit:
            results_data = results_data[:limit]
            
        processed_results = []
        if progress_bar:
            results_data = tqdm(
                results_data,
                desc="Post-Processing CoT results",
                unit="result"
            )
        
        for result in results_data:
            # Deserialize the SearchResult
            strategy_registry = self.get_strategy_registry()
            search_result = SearchResult.deserialize(result["result"], strategy_registry)
            
            # Get question and ground truth
            question = result["question"]
            ground_truth = result["ground_truth"]
            
            # Get the nodes to process. If only_successful is True, we only process successful nodes.
            # Otherwise, we process all terminal nodes.
            nodes_to_process = (
                search_result.get_successful_terminal_nodes()
                if only_successful
                else search_result.terminal_nodes
            )
            for node in nodes_to_process:
                cot = node.get_full_cot()
                
                # Generate natural reasoning
                response_dict, error = self.process(question=question, cot=cot)
                if error:
                    logger.warning(f"Error generating natural reasoning: {error}")
                    continue
                natural_reasoning = response_dict.get("natural_reasoning")
                formal_response = response_dict.get("formal_response")

                # Create processed result
                processed_result = {
                    "id": result["id"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "natural_reasoning": natural_reasoning,
                    "formal_response": formal_response,
                }
                
                # Add to results list
                processed_results.append(processed_result)
                
                # Save to file
                self._save_processed_result(processed_result)
            
        return processed_results
    
    def _save_processed_result(self, result: dict[str, Any]) -> None:
        """Save a processed result to the processed results file."""
        
        with open(self.output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    def load_processed_results(self) -> list[dict[str, Any]]:
        """
        Load all processed results from disk.
        
        Returns:
            List of processed result dictionaries
        """
        
        results = []
        if not self.output_path.exists():
            return results
        
        with open(self.output_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        return results
        
    def generate_natural_reasoning(
        self,
        question: str,
        cot: dict,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        llm_kwargs: dict[str, Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Generate natural reasoning from a CoT.

        Args:
            question (str): The question to be answered.
            cot (dict): The chain of thought to be reformatted.
            on_error (Literal): Error handling strategy.
            llm_kwargs (dict): Additional arguments for the LLM provider.
            max_retries (int): Maximum number of retries for failed requests.
            retry_delay (float): Delay between retries in seconds.

        Returns:
            str: The generated natural reasoning.
        """
        llm_kwargs = llm_kwargs or {}
        
        # Build the natural language CoT prompt
        prompt = build_natural_language_cot_prompt(
            question=question,
            cot=cot,
        )
        
        # Generate and parse reasoning using the LLM
        response, natural_reasoning = generate_and_parse_json(
            llm_provider=self.llm_provider,
            prompt=prompt,
            on_error=on_error,
            llm_kwargs=llm_kwargs,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retrieval_object="NaturalReasoning"
        )
        
        if not response or not natural_reasoning:
            logger.error(
                f"Failed to generate natural reasoning from the CoT: {response}."
            )
            return ""
            
        return natural_reasoning
            
        
    def generate_formal_response(
        self,
        question: str,
        natural_reasoning: str
    ) -> str:
        """
        Generate a formal response based on the natural reasoning.
        Args:
            question (str): The question to be answered.
            natural_reasoning (str): The natural reasoning process.
        Returns:
            str: The formatted formal response.
        """
        # Build the formal response prompt
        prompt = build_formal_response_prompt(
            question=question,
            natural_reasoning=natural_reasoning,
        )
        
        # Generate and return the formal response using the LLM
        return self.llm_provider.generate(prompt)
    
    def process(
        self,
        question: str,
        cot: dict,
        on_error: Literal["continue", "raise", "retry"] = "retry",
        llm_kwargs: dict[str, Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> tuple[dict, str | None]:
        """
        Process the CoT to generate natural reasoning and formal response.

        Args:
            question (str): The question to be answered.
            cot (dict): The chain of thought to be reformatted.
            on_error (Literal): Error handling strategy.
            llm_kwargs (dict): Additional arguments for the LLM provider.
            max_retries (int): Maximum number of retries for failed requests.
            retry_delay (float): Delay between retries in seconds.

        Returns:
            tuple[dict, str | None]: The generated reasoning and any error message.
        """
        # Generate natural reasoning
        natural_reasoning = self.generate_natural_reasoning(
            question=question,
            cot=cot,
            on_error=on_error,
            llm_kwargs=llm_kwargs,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        if not natural_reasoning:
            return {}, "Failed to generate natural reasoning."
        
        # Generate formal response
        formal_response = self.generate_formal_response(
            question=question,
            natural_reasoning=natural_reasoning
        )
        
        return {"natural_reasoning": natural_reasoning, "formal_response": formal_response}, None