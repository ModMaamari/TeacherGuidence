"""
Base component class that all workflow components inherit from.

This provides a common interface and metadata structure for all components.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger


class ComponentCategory(str, Enum):
    """Categories for organizing components"""
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    PLANNING = "planning"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    CONTROL = "control"


@dataclass
class ComponentResult:
    """
    Standardized result from component execution.
    
    Attributes:
        success: Whether the component executed successfully
        data: Output data from the component
        metadata: Additional metadata about the execution
        error: Error message if execution failed
        execution_time_ms: Time taken to execute in milliseconds
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ComponentSpec:
    """
    Specification for a component, defining its interface and requirements.
    
    Attributes:
        name: Component name
        category: Component category
        description: Human-readable description
        input_keys: Required input keys from context
        output_keys: Keys that will be written to context
        config_schema: JSON schema for configuration validation
        requires_llm: Whether this component requires LLM access
    """
    name: str
    category: ComponentCategory
    description: str
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    requires_llm: bool = False


class BaseComponent(ABC):
    """
    Abstract base class for all workflow components.
    
    Each component:
    1. Reads input from WorkflowContext
    2. Performs some operation
    3. Writes output back to WorkflowContext
    4. Returns a ComponentResult with execution details
    
    Components are designed to be:
    - Stateless: All state lives in WorkflowContext
    - Composable: Can be chained together in workflows
    - Testable: Clear inputs and outputs
    - Extensible: Easy to create new components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize component with configuration.
        
        Args:
            config: Component-specific configuration
        """
        self.config = config or {}
        self._validate_config()
    
    @property
    @abstractmethod
    def spec(self) -> ComponentSpec:
        """
        Return the component specification.
        
        This defines the component's interface and requirements.
        """
        pass
    
    @abstractmethod
    async def execute(self, context: Any) -> ComponentResult:
        """
        Execute the component logic.
        
        Args:
            context: WorkflowContext containing shared state
            
        Returns:
            ComponentResult with execution details
        """
        pass
    
    def _validate_config(self) -> None:
        """
        Validate component configuration against schema.
        
        Override this method to add custom validation logic.
        """
        # Basic validation - can be extended by subclasses
        schema = self.spec.config_schema
        
        for key, schema_def in schema.items():
            if key not in self.config and "default" in schema_def:
                self.config[key] = schema_def["default"]
    
    async def __call__(self, context: Any) -> ComponentResult:
        """
        Make component callable for convenience.
        
        This allows: result = await component(context)
        """
        return await self.execute(context)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(category={self.spec.category.value})"


class ComponentRegistry:
    """
    Registry for discovering and instantiating components.
    
    This allows workflows to reference components by name
    and automatically instantiate them with the correct configuration.
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a component class.
        
        Usage:
            @ComponentRegistry.register("opensearch_retriever")
            class OpenSearchRetriever(BaseComponent):
                ...
        """
        def decorator(component_class: type):
            cls._registry[name] = component_class
            logger.debug(f"Registered component: {name} -> {component_class.__name__}")
            return component_class
        return decorator
    
    @classmethod
    def get(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseComponent:
        """
        Get a component instance by name.
        
        Args:
            name: Component name
            config: Component configuration
            
        Returns:
            Instantiated component
            
        Raises:
            KeyError: If component not found
        """
        if name not in cls._registry:
            raise KeyError(f"Component '{name}' not found in registry. "
                          f"Available: {list(cls._registry.keys())}")
        
        component_class = cls._registry[name]
        return component_class(config=config)
    
    @classmethod
    def list_components(cls, category: Optional[ComponentCategory] = None) -> List[str]:
        """
        List all registered components, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of component names
        """
        if category is None:
            return list(cls._registry.keys())
        
        # Filter by category
        result = []
        for name, component_class in cls._registry.items():
            # Instantiate temporarily to get spec
            temp_instance = component_class()
            if temp_instance.spec.category == category:
                result.append(name)
        
        return result
    
    @classmethod
    def get_spec(cls, name: str) -> ComponentSpec:
        """
        Get component specification without instantiating.
        
        Args:
            name: Component name
            
        Returns:
            ComponentSpec
        """
        component_class = cls._registry[name]
        temp_instance = component_class()
        return temp_instance.spec

