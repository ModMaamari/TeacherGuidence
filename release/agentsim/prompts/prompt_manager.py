"""
Prompt template manager for loading and rendering LLM prompts.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class PromptManager:
    """
    Manages loading and rendering of prompt templates.
    
    Supports:
    - Template variables with {variable} syntax
    - Multi-line prompts
    - Organized by component category
    - Easy versioning and A/B testing
    
    Example:
        >>> pm = PromptManager()
        >>> prompt = pm.render("planning/decompose", query="What causes earthquakes?", max_subgoals=5)
    """
    
    _instance = None
    _templates_cache: Dict[str, str] = {}
    
    def __new__(cls):
        """Singleton pattern to cache loaded templates"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize prompt manager"""
        if self._initialized:
            return
            
        self.prompts_dir = Path(__file__).parent.parent.parent / "templates" / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.debug(f"PromptManager initialized with dir: {self.prompts_dir}")
    
    def load_template(self, template_name: str) -> str:
        """
        Load a prompt template from file.
        
        Args:
            template_name: Template path (e.g., "planning/decompose")
            
        Returns:
            Template content as string
            
        Raises:
            FileNotFoundError: If template doesn't exist
        """
        # Check cache first
        if template_name in self._templates_cache:
            return self._templates_cache[template_name]
        
        # Try with .txt extension
        template_path = self.prompts_dir / f"{template_name}.txt"
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_name}\n"
                f"Expected at: {template_path}"
            )
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cache the template
            self._templates_cache[template_name] = content
            logger.debug(f"Loaded prompt template: {template_name}")
            return content
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    def render(self, template_name: str, **variables) -> str:
        """
        Load and render a prompt template with variables.
        
        Args:
            template_name: Template path (e.g., "planning/decompose")
            **variables: Variables to substitute in the template
            
        Returns:
            Rendered prompt string
            
        Example:
            >>> pm.render("planning/decompose", query="Test", max_subgoals=3)
        """
        template = self.load_template(template_name)
        
        try:
            # Simple variable substitution using str.format()
            rendered = template.format(**variables)
            return rendered
            
        except KeyError as e:
            missing_var = str(e).strip("'")
            logger.error(
                f"Missing variable '{missing_var}' for template '{template_name}'. "
                f"Provided: {list(variables.keys())}"
            )
            raise ValueError(
                f"Template '{template_name}' requires variable '{missing_var}' "
                f"but it was not provided"
            ) from e
    
    def list_templates(self, category: Optional[str] = None) -> list[str]:
        """
        List available prompt templates.
        
        Args:
            category: Optional category filter (e.g., "planning")
            
        Returns:
            List of template names
        """
        templates = []
        
        search_dir = self.prompts_dir / category if category else self.prompts_dir
        
        if not search_dir.exists():
            return templates
        
        for template_file in search_dir.rglob("*.txt"):
            # Get relative path from prompts_dir
            rel_path = template_file.relative_to(self.prompts_dir)
            # Remove .txt extension
            template_name = str(rel_path).replace('.txt', '')
            templates.append(template_name)
        
        return sorted(templates)
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get metadata about a template (variables, size, etc).
        
        Args:
            template_name: Template path
            
        Returns:
            Dictionary with template metadata
        """
        template = self.load_template(template_name)
        
        # Extract variable names using regex
        variables = re.findall(r'\{(\w+)\}', template)
        unique_vars = sorted(set(variables))
        
        return {
            "name": template_name,
            "size_bytes": len(template.encode('utf-8')),
            "line_count": template.count('\n') + 1,
            "required_variables": unique_vars,
            "char_count": len(template)
        }
    
    def clear_cache(self):
        """Clear the template cache (useful for development/testing)"""
        self._templates_cache.clear()
        logger.debug("Template cache cleared")

