"""
Workflow loader for YAML-based workflow definitions.

This module loads workflow templates from YAML files and
instantiates the component pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from agentsim.components.base import ComponentRegistry


@dataclass
class WorkflowDefinition:
    """
    Parsed workflow definition from YAML.
    
    Attributes:
        id: Workflow identifier
        name: Human-readable name
        description: Workflow description
        reasoning_style: Reasoning approach
        components: List of component configurations
        config: Global workflow configuration
    """
    id: str
    name: str
    description: str
    reasoning_style: str
    components: List[Dict[str, Any]]
    config: Dict[str, Any]


class WorkflowLoader:
    """
    Loads and parses workflow definitions from YAML files.
    
    YAML Format:
    ```yaml
    id: gpt4o_style
    name: GPT-4o Style
    description: Breadth-first exploration workflow
    reasoning_style: breadth_first
    
    components:
      - type: planner
        config:
          max_subgoals: 5
          temperature: 0.7
      
      - type: opensearch_retriever
        config:
          k: 50
          multi_angle: true
    
    config:
      max_iterations: 5
      answer_threshold: 0.6
    ```
    """
    
    def __init__(self, workflows_dir: Optional[Path] = None):
        """
        Initialize workflow loader.
        
        Args:
            workflows_dir: Directory containing workflow YAML files
        """
        if workflows_dir is None:
            # Default to templates/workflows at package root
            self.workflows_dir = Path(__file__).resolve().parent.parent.parent / "templates" / "workflows"
        else:
            self.workflows_dir = Path(workflows_dir)
        
        logger.info(f"WorkflowLoader initialized with workflows_dir: {self.workflows_dir}")
    
    def load_workflow(self, workflow_id: str) -> WorkflowDefinition:
        """
        Load a workflow by ID.
        
        Args:
            workflow_id: Workflow identifier (e.g., "gpt4o_style")
            
        Returns:
            WorkflowDefinition
            
        Raises:
            FileNotFoundError: If workflow template not found
            ValueError: If workflow definition is invalid
        """
        template_path = self.workflows_dir / f"{workflow_id}.yaml"
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Workflow template not found: {template_path}. "
                f"Available: {self.list_available_workflows()}"
            )
        
        logger.info(f"Loading workflow from: {template_path}")
        
        with open(template_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return self._parse_workflow(data)
    
    def load_workflow_from_dict(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """
        Load workflow from a dictionary (e.g., from API request).
        
        Args:
            data: Workflow definition as dictionary
            
        Returns:
            WorkflowDefinition
        """
        return self._parse_workflow(data)
    
    def _parse_workflow(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """
        Parse workflow data into WorkflowDefinition.
        
        Args:
            data: Raw workflow data
            
        Returns:
            WorkflowDefinition
            
        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        required = ["id", "name", "components"]
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate components
        components = data.get("components", [])
        if not isinstance(components, list) or not components:
            raise ValueError("Workflow must have at least one component")
        
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                raise ValueError(f"Component {i} must be a dictionary")
            if "type" not in comp:
                raise ValueError(f"Component {i} missing 'type' field")
        
        return WorkflowDefinition(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            reasoning_style=data.get("reasoning_style", "default"),
            components=components,
            config=data.get("config", {})
        )
    
    def list_available_workflows(self) -> List[str]:
        """
        List all available workflow templates.
        
        Returns:
            List of workflow IDs
        """
        if not self.workflows_dir.exists():
            return []
        
        workflows = []
        for template_file in self.workflows_dir.glob("*.yaml"):
            workflows.append(template_file.stem)
        
        return sorted(workflows)
    
    def get_workflow_info(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get workflow metadata without full loading.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dictionary with workflow metadata
        """
        try:
            workflow = self.load_workflow(workflow_id)
            return {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "reasoning_style": workflow.reasoning_style,
                "component_count": len(workflow.components),
                "component_types": [c.get("type") for c in workflow.components]
            }
        except Exception as e:
            return {"id": workflow_id, "error": str(e)}
    
    def list_workflows_with_info(self) -> List[Dict[str, Any]]:
        """
        List all workflows with their metadata.
        
        Returns:
            List of workflow info dictionaries
        """
        workflows = []
        for workflow_id in self.list_available_workflows():
            workflows.append(self.get_workflow_info(workflow_id))
        return workflows
    
    def validate_workflow(self, workflow: WorkflowDefinition) -> tuple[bool, List[str]]:
        """
        Validate a workflow definition.
        
        Args:
            workflow: WorkflowDefinition to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check that all component types are registered
        for i, comp_def in enumerate(workflow.components):
            comp_type = comp_def.get("type")
            try:
                ComponentRegistry.get_spec(comp_type)
            except KeyError:
                errors.append(f"Component {i}: Unknown type '{comp_type}'")
        
        # Validate component configs against schemas
        for i, comp_def in enumerate(workflow.components):
            comp_type = comp_def.get("type")
            comp_config = comp_def.get("config", {})
            
            try:
                spec = ComponentRegistry.get_spec(comp_type)
                schema = spec.config_schema
                
                # Basic schema validation
                for key, value in comp_config.items():
                    if key not in schema:
                        errors.append(
                            f"Component {i} ({comp_type}): Unknown config key '{key}'"
                        )
            except KeyError:
                pass  # Already caught above
        
        return len(errors) == 0, errors

