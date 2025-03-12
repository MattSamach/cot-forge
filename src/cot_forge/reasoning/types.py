from typing import Any, Optional, TypedDict

from cot_forge.reasoning.strategies import Strategy


class ReasoningNode:
    """A node in the reasoning graph/tree/chain."""
    def __init__(self, 
                 strategy: Strategy | None,
                 prompt: str,
                 response: str,
                 cot: dict[str, Any] | None = None,
                 parent: Optional['ReasoningNode'] = None):
        self.strategy = strategy
        self.prompt = prompt
        self.response = response
        self.cot: dict[str, Any] = cot
        self.parent = parent
        self.children: list[ReasoningNode] = []
        self.is_final = False
        self.metadata: dict[str, Any] = {}

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
    
    def get_full_cot(self):
        """Get the complete chain of thought (CoT) from the root to this node."""
        chain = []
        current_node = self
        while current_node:
            if current_node.cot:
                chain.extend(current_node.cot)
            current_node = current_node.parent
        return list(chain)
    
    def __repr__(self):
        return f"ReasoningNode(strategy={self.strategy}, prompt={self.prompt}, response={self.response})"
    
    def __str__(self):
        return f"ReasoningNode(strategy={self.strategy}, prompt={self.prompt}, cot={self.cot}, parent={self.parent})"
    
    

class SearchResult(TypedDict):
    """Represents the result of a search algorithm."""
    final_node: ReasoningNode | None
    all_terminal_nodes: list[ReasoningNode] | None
    success: bool
    final_answer: Optional[str]
    metadata: dict[str, Any]