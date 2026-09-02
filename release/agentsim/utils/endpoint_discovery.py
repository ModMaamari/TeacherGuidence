"""Discover available models from LLM endpoints"""

import httpx
from typing import List, Dict, Optional
from loguru import logger

from agentsim.config import config


async def discover_custom_endpoint_models(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    """Discover available models from custom endpoint
    
    Args:
        endpoint: Custom endpoint URL (defaults to CUSTOM_LLM_ENDPOINT)
        api_key: API key (defaults to CUSTOM_LLM_API_KEY)
    
    Returns:
        List of models with their IDs and metadata
    """
    endpoint = endpoint or config.CUSTOM_LLM_ENDPOINT
    api_key = api_key or config.CUSTOM_LLM_API_KEY
    
    if not endpoint:
        logger.error("No custom endpoint configured")
        return []
    
    # Try OpenAI-compatible /v1/models endpoint
    models_url = endpoint.rstrip('/') + '/v1/models'
    
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(models_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response
            if 'data' in data:
                models = []
                for model in data['data']:
                    models.append({
                        'id': model.get('id', 'unknown'),
                        'name': model.get('id', 'unknown'),
                        'created': model.get('created'),
                        'owned_by': model.get('owned_by', 'unknown')
                    })
                return models
            elif 'models' in data:
                # Alternative format
                return data['models']
            else:
                logger.warning("Unexpected response format")
                return []
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error querying endpoint: {e}")
        return []
    except Exception as e:
        logger.error(f"Error discovering models: {e}")
        return []


async def discover_ollama_models(
    endpoint: Optional[str] = None
) -> List[Dict[str, str]]:
    """Discover available Ollama models
    
    Args:
        endpoint: Ollama endpoint URL (defaults to OLLAMA_ENDPOINT)
    
    Returns:
        List of models with their IDs and metadata
    """
    endpoint = endpoint or config.OLLAMA_ENDPOINT
    
    if not endpoint:
        logger.error("No Ollama endpoint configured")
        return []
    
    tags_url = endpoint.rstrip('/') + '/api/tags'
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(tags_url)
            response.raise_for_status()
            
            data = response.json()
            
            if 'models' in data:
                models = []
                for model in data['models']:
                    models.append({
                        'id': model.get('name', 'unknown'),
                        'name': model.get('name', 'unknown'),
                        'size': model.get('size'),
                        'modified_at': model.get('modified_at')
                    })
                return models
            else:
                return []
                
    except Exception as e:
        logger.error(f"Error discovering Ollama models: {e}")
        return []


def format_models_table(models: List[Dict[str, str]], provider: str) -> str:
    """Format models as a readable table"""
    if not models:
        return f"No models found for {provider}"
    
    lines = [f"\n{provider} Models:", "=" * 60]
    
    for model in models:
        model_id = model.get('id', model.get('name', 'unknown'))
        lines.append(f"  • {model_id}")
        
        if 'owned_by' in model and model['owned_by']:
            lines.append(f"    Owned by: {model['owned_by']}")
        if 'size' in model and model['size']:
            size_gb = model['size'] / (1024**3)
            lines.append(f"    Size: {size_gb:.2f} GB")
    
    lines.append("")
    return "\n".join(lines)

