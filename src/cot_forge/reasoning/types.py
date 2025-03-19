from typing import Any, Optional, TypedDict

from cot_forge.reasoning.strategies import Strategy


class ReasoningNode:
    """A node in the reasoning graph/tree/chain."""
    def __init__(self, 
             strategy: Strategy | None,
             prompt: str,
             response: str,
             cot: list[dict[str, Any]] | None = None,  # Changed from dict to list[dict]
             parent: Optional['ReasoningNode'] = None,
             metadata: dict[str, Any] = None,
             ):
        self.strategy = strategy
        self.prompt = prompt
        self.response = response
        self.cot: dict[str, Any] = cot
        self.parent = parent
        self.children: list[ReasoningNode] = []
        self.is_final = False
        self.success = False
        self.metadata = {} if metadata is None else metadata

    def add_child(self, child: 'ReasoningNode'):
        self.children.append(child)
        
    def get_full_node_chain(self):
        """Get the complete chain from the root to this node."""
        chain = []
        current_node = self
        while current_node:
            chain.append(current_node)
            current_node = current_node.parent
        return list(reversed(chain))
    
    def get_full_cot(self) -> list[dict[str, Any]]:
        """Get the complete chain of thought (CoT) from the root to this node."""
        nodes = self.get_full_node_chain()  # Already in root-to-current order
        result = []
        for node in nodes:
            if node.cot:
                result.extend(node.cot)
        return result
    
    def __repr__(self):
        return (f"ReasoningNode(strategy={self.strategy}, ",
                f"prompt={self.prompt}, response={self.response}, "
                f"cot={self.cot}, parent={self.parent}, "
                f"children={self.children}, is_final={self.is_final}, "
                f"success={self.success}, metadata={self.metadata})")
    
    def __str__(self):
        return (f"ReasoningNode(strategy={self.strategy}, ",
                f"prompt={self.prompt}, response={self.response}, "
                f"cot={self.cot}, parent={self.parent}, "
                f"children={self.children}, is_final={self.is_final}, "
                f"success={self.success}, metadata={self.metadata})")

class SearchResult:
    """Represents the result of a search algorithm."""
    
    def __init__(
        self,
        question: str = "",
        ground_truth_answer: str = "",
        final_node: ReasoningNode | None = None,
        all_terminal_nodes: list[ReasoningNode] | None = None,
        success: bool = False,
        final_answer: Optional[str] = None,
        metadata: dict[str, Any] = None
    ):
        self.final_node = final_node
        self.question = question
        self.ground_truth_answer = ground_truth_answer
        self.all_terminal_nodes = all_terminal_nodes if all_terminal_nodes else []
        self.success = success
        self.final_answer = final_answer
        self.metadata = metadata if metadata else {}
