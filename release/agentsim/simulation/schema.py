"""Simulation template schema"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SimulationMode(str, Enum):
    STANDARD = "standard"  # Fixed workflow iteration
    ADAPTIVE = "adaptive"  # Adaptive component selection
    EXPLORATORY = "exploratory"  # Exploratory expansion


class SimilarityMetric(str, Enum):
    EMBEDDING_COSINE = "embedding_cosine"
    TOKEN_OVERLAP = "token_overlap"
    BLEU = "bleu"
    ROUGE = "rouge"


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    model_id: str  # e.g., "gpt-4o", "claude-sonnet-3.5"
    role: str  # "teacher", "consultant", "verifier"
    api_key: Optional[str] = None
    temperature: float = 0.7


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    path: str
    num_samples: int = -1  # -1 = all
    sample_strategy: str = "sequential"  # sequential, random, stratified


@dataclass
class ParallelConfig:
    """Parallel execution for specific components"""
    component_type: str  # e.g., "synthesis", "planner"
    consultant_models: List[str]  # Model names to run in parallel


@dataclass
class VerificationConfig:
    """Real-time verification configuration"""
    enabled: bool = True
    verifier_model: str = "gpt-4o-mini"
    check_synthesis: bool = True
    check_claims: bool = True
    flag_hallucinations: bool = True


@dataclass
class ExploratoryConfig:
    """Exploratory mode-specific configuration"""
    max_explorations: int = 10  # Number of exploration iterations
    use_knowledge_base: bool = True  # Whether to use growing knowledge base
    execution_mode: str = "standard"  # "standard", "adaptive", or "auto" (teacher chooses)
    reflection_temperature: float = 0.8  # Temperature for reflection/question generation
    questions_per_reflection: int = 3  # How many new questions to generate
    external_knowledge_component: Optional[str] = None  # Component for external knowledge check
    enable_consultant_reflection: bool = True  # Allow consultants to suggest questions


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    type: str = "chatnoir"  # chatnoir, opensearch, vector, hybrid
    enabled: bool = True
    corpus: Optional[str] = None  # For ChatNoir: cw12, msmarco, etc.
    top_k: int = 50
    base_url: Optional[str] = None  # Override from env
    api_key: Optional[str] = None  # Override from env
    index: Optional[str] = None  # For OpenSearch
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationTemplate:
    """Complete simulation configuration"""
    id: str
    name: str
    mode: SimulationMode
    teacher_models: List[ModelConfig]
    workflows: List[str]
    
    # Optional fields
    consultant_models: List[ModelConfig] = field(default_factory=list)
    verifier_model: Optional[ModelConfig] = None
    datasets: List[DatasetConfig] = field(default_factory=list)
    retrieval: Optional[RetrievalConfig] = None
    
    # Execution
    max_iterations: int = 5
    similarity_metric: SimilarityMetric = SimilarityMetric.EMBEDDING_COSINE
    similarity_threshold: float = 0.6
    
    # Parallel execution
    parallel_execution: List[ParallelConfig] = field(default_factory=list)
    
    # Verification
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    
    # Output
    output_dir: str = "./data/simulation_output"
    
    # Mode-specific config
    mode_config: Dict[str, Any] = field(default_factory=dict)

