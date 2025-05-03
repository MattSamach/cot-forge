"""
This module provides persistence functionality for CoTBuilder.

It handles checkpointing, resumable operations, and result storage for CoT generations
to ensure that long-running jobs can be interrupted and resumed without data loss.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cot_forge.reasoning.cot_builder import CoTBuilder

from cot_forge.reasoning.types import SearchResult

logger = logging.getLogger(__name__)

class PersistenceManager:
    """
    Manages persistence for CoTBuilder operations.
    
    Responsible for:
    1. Saving and loading CoTBuilder configurations
    2. Tracking progress of batch operations
    3. Handling SearchResults data
    4. Handling reasoning data from reasoning_processor
    
    Attributes:
        dataset_name (str): Unique identifier for the dataset/run
        base_dir (Path): Base directory for storing data
        search_dir (Path): Directory for this specific dataset
        config_path (Path): Path to config JSON file
        search_results_path (Path): Path to results JSONL file
        reasoning_path (Path): Path to reasoning JSONL file
        metadata_path (Path): Path to metadata JSON file
    """
    
    def __init__(
        self, 
        dataset_name: str,
        search_name: str,
        base_dir: str = "data",
    ):
        """
        Initialize a persistence manager for CoTBuilder.
        
        Args:
            dataset_name: Unique identifier for this dataset/run
            search_name: Name of the search algorithm
            base_dir: Base directory for data storage
        """
        self.dataset_name = dataset_name
        self.search_name = search_name
        self.base_dir = Path(base_dir)
        self.search_dir = self.base_dir / dataset_name / search_name
        
        # File paths
        self.config_path = self.search_dir / "config.json"
        self.search_results_path = self.search_dir / "results.jsonl"
        self.reasoning_path = self.search_dir / "reasoning.jsonl"
        self.metadata_path = self.search_dir / "metadata.json"
        
        # Progress tracking
        self.processed_ids = set()
        self.completed_items = 0
        self.successful_items = 0
        
        # Thread safety
        self._lock = RLock()
        
        # Initialize directories and files
        self._initialize_storage()
        
        # Load metadata to track progress
        self.load_metadata()

    def _initialize_storage(self):
        """Create necessary directories and files if they don't exist."""
        # Create directories
        self.search_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file if it doesn't exist
        if not self.metadata_path.exists():
            self._save_metadata()
            
    def generate_question_id(self, question: str, ground_truth: str) -> str:
        """Generate a unique ID for a question-answer pair."""
        import hashlib

        # Create a hash from the question and answer
        return hashlib.md5(f"{question}:{ground_truth}".encode()).hexdigest()
    
    def save_config(self, cot_builder: 'CoTBuilder') -> None:
        """
        Save the CoTBuilder configuration.
        
        Args:
            cot_builder: The CoTBuilder instance to save
        """

        config = {
            "search_llm": cot_builder.search_llm.to_dict(),
            "post_processing_llm": cot_builder.post_processing_llm.to_dict(),
            "search": cot_builder.search.to_dict(),
            "verifier": cot_builder.verifier.to_dict(),
            "scorer": cot_builder.scorer.to_dict() if cot_builder.scorer else None,
            "strategy_registry": cot_builder.strategy_registry.serialize(),
            "search_llm_kwargs": cot_builder.search_llm_kwargs,
            "post_processing_llm_kwargs": cot_builder.post_processing_llm_kwargs,
            "created_at": datetime.now().isoformat(),
            "dataset_name": self.dataset_name,
            "search_name": self.search_name,
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved CoTBuilder configuration to {self.config_path}")
        
    def load_config(self) -> dict[str, Any]:
        """
        Load the CoTBuilder configuration from disk.
        
        Returns:
            dict: The loaded configuration
        """
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} does not exist.")
            return {}
        
        with open(self.config_path) as f:
            config = json.load(f)
        
        logger.info(f"Loaded CoTBuilder configuration from {self.config_path}")
        return config
    
    def _save_metadata(self) -> None:
        """Save current progress metadata."""
        # Check if file at metadata path already exists
        if self.metadata_path.exists():
            return
        # Create metadata directory if it doesn't exist
        with self._lock:
            metadata = {
                "dataset_name": self.dataset_name,
                "search_name": self.search_name,
                "completed_items": self.completed_items,
                "successful_items": self.successful_items,
                "processed_ids": list(self.processed_ids),
                "last_updated": datetime.now().isoformat(),
            }
        
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> bool:
        """
        Load metadata from disk.
        
        Returns:
            bool: True if metadata was loaded, False otherwise
        """
        if not self.metadata_path.exists():
            return False
        
        try:
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            
            self.completed_items = metadata.get("completed_items", 0)
            self.successful_items = metadata.get("successful_items", 0)
            self.processed_ids = set(metadata.get("processed_ids", []))
            
            logger.info(f"Loaded metadata: {self.completed_items} completed")
            return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def should_skip(self, question: str, ground_truth: str) -> bool:
        """
        Check if this question-answer pair has already been processed.
        
        Args:
            question: The question text
            ground_truth: The ground truth answer
            
        Returns:
            bool: True if this pair should be skipped (already processed)
        """
        question_id = self.generate_question_id(question, ground_truth)
        return question_id in self.processed_ids
    
    def save_result(
        self,
        result: SearchResult,
        reasoning: dict,
        question: str,
        ground_truth: str
    ) -> None:
        """
        Save a search result to the results file.
        
        Args:
            result: The SearchResult to save
            reasoning: The reasoning data from the reasoning_processor
            question: The question text
            ground_truth: The ground truth answer
        """
        question_id = self.generate_question_id(question, ground_truth)
        
        # Prepare the result data
        search_data = {
            "id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "success": result.success,
            "timestamp": datetime.now().isoformat(),
            "result": result.serialize() if hasattr(result, "serialize") else str(result)
        }
        
        # Update data
        with self._lock:
            # Append to the results file
            with open(self.search_results_path, 'a') as f:
                f.write(json.dumps(search_data) + '\n')
                
            # Append reasoning data
            with open(self.reasoning_path, 'a') as f:
                f.write(json.dumps(reasoning) + '\n')
        
            # Update tracking info
            self.processed_ids.add(question_id)
            self.completed_items += 1
            if result.success:
                self.successful_items += 1
            
            # Save metadata after each result
            self._save_metadata()

        logger.info(f"Saved result for {question_id}: {result.success}")
    
    def load_search_results(self) -> list[dict[str, Any]]:
        """
        Load search results from the results file.
        
        Returns:
            List of result data dictionaries
        """
        results = []
        if not self.search_results_path.exists():
            return results
        
        with open(self.search_results_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        return results
    
    def load_reasoning(self) -> list[dict[str, Any]]:
        """
        Load reasoning data from the reasoning file.
        
        Returns:
            List of reasoning data dictionaries
        """
        reasoning = []
        if not self.reasoning_path.exists():
            return reasoning
        
        with open(self.reasoning_path) as f:
            for line in f:
                if line.strip():
                    reasoning.append(json.loads(line))
        
        return reasoning
    
    def setup_batch_run(self) -> None:
        """
        Set up for a batch run by initializing total items.

        """
        with self._lock:
            self._save_metadata()
            
    def reset_all_files(self) -> None:
        """
        Delete all files related to this dataset and search, then reinitialize storage.
        """
        with self._lock:
            if self.search_dir.exists():
                for file in self.search_dir.iterdir():
                    file.unlink()
                self.search_dir.rmdir()
                logger.info(f"Deleted all files in {self.search_dir}")
            else:
                logger.warning(f"Directory {self.search_dir} does not exist.")
            
            # Reset tracking info
            self.processed_ids = set()
            self.completed_items = 0
            self.successful_items = 0
            
            # Reinitialize storage
            self._initialize_storage()
            logger.info(f"Reinitialized storage in {self.search_dir}")
            
    def delete_search_results(self, question: str, ground_truth: str) -> None:
        """
        Delete results for a specific question-answer pair.
        
        Args:
            question: The question text
            ground_truth: The ground truth answer
        """
        question_id = self.generate_question_id(question, ground_truth)
        
        # Rebuild results file without this entry
        results = self.load_search_results()
        
        # Count what will be deleted and filter the list
        deleted_items = [r for r in results if r["id"] == question_id]
        num_results_deleted = len(deleted_items)
        num_successes_deleted = sum(1 for r in deleted_items if r["success"])
        
        # Create new list without the matching items
        filtered_results = [r for r in results if r["id"] != question_id]
                
        # Write back the updated results
        with open(self.search_results_path, 'w') as f:
            for result in filtered_results:
                f.write(json.dumps(result) + '\n')
        
        # Free memory
        del results
        del filtered_results
        del deleted_items
        
        # Load reasoning data
        reasoning = self.load_reasoning()
        # Filter out reasoning data for the deleted results
        filtered_reasoning = [r for r in reasoning if r.get("id") != question_id]
        # Write back the updated reasoning data
        with open(self.reasoning_path, 'w') as f:
            for r in filtered_reasoning:
                f.write(json.dumps(r) + '\n')
        # Free memory
        del reasoning
        del filtered_reasoning
                
        logger.info(f"Deleted {num_results_deleted} result(s) for {question_id}")
        
        # Update metadata
        self.processed_ids.discard(question_id)
        self.completed_items -= num_results_deleted
        self.successful_items -= num_successes_deleted
            
        # Save updated metadata
        self._save_metadata()