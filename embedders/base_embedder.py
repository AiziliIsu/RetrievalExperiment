"""
Base Embedder Interface
All embedding models must implement this interface for consistent testing.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedder with configuration.
        
        Args:
            config: Model configuration from config.yaml
        """
        self.name = config['name']
        self.model_id = config['model_id']
        self.dimension = config['dimension']
        self.batch_size = config.get('batch_size', 32)
        self.max_tokens = config.get('max_tokens', 512)
        
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            numpy array of shape (dimension,)
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            'name': self.name,
            'model_id': self.model_id,
            'dimension': self.dimension,
            'batch_size': self.batch_size,
            'max_tokens': self.max_tokens
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.dimension})"
