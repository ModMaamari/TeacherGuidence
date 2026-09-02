"""Load and validate simulation templates"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

from agentsim.simulation.schema import (
    SimulationTemplate, SimulationMode, SimilarityMetric,
    ModelConfig, DatasetConfig, ParallelConfig, VerificationConfig, RetrievalConfig
)
from agentsim.workflow.loader import WorkflowLoader
from agentsim.config import config


class SimulationLoader:
    """Load simulation templates with validation"""
    
    def __init__(self, templates_dir: Path = None):
        if templates_dir is None:
            # Default to templates/simulations at package root
            self.templates_dir = Path(__file__).resolve().parent.parent.parent / "templates" / "simulations"
        else:
            self.templates_dir = Path(templates_dir)
        self.workflow_loader = WorkflowLoader()
    
    def load(self, simulation_id: str) -> SimulationTemplate:
        """Load simulation template"""
        path = self.templates_dir / f"{simulation_id}.yaml"
        
        if not path.exists():
            raise FileNotFoundError(f"Simulation template not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return self._parse(data)
    
    def _parse(self, data: Dict[str, Any]) -> SimulationTemplate:
        """Parse simulation data with environment variable fallbacks"""
        
        # Parse models - use env vars if not specified in YAML
        teacher_models_data = data.get("teacher_models", [])
        
        if not teacher_models_data:
            # Create from environment (supports multiple models)
            teacher_models = []
            for i, model_id in enumerate(config.TEACHER_MODELS):
                provider = config.get_provider_from_model_id(model_id)
                teacher_models.append(ModelConfig(
                    name=f"{provider}_{i}" if len(config.TEACHER_MODELS) > 1 else "teacher",
                    model_id=model_id,
                    role="teacher",
                    api_key=config.get_model_api_key(model_id),
                    temperature=config.TEACHER_TEMPERATURE
                ))
        else:
            # Parse from YAML, inject API keys from env if not provided
            teacher_models = []
            for m in teacher_models_data:
                if "api_key" not in m or not m["api_key"]:
                    m["api_key"] = config.get_model_api_key(m.get("model_id", ""))
                teacher_models.append(ModelConfig(**m))
        
        # Parse consultant models - use env vars if available
        consultant_models_data = data.get("consultant_models", [])
        
        if not consultant_models_data and config.CONSULTANT_MODELS:
            # Create from environment
            consultant_models = []
            for model_id in config.CONSULTANT_MODELS:
                if model_id.strip():
                    consultant_models.append(ModelConfig(
                        name=model_id.split("-")[0],
                        model_id=model_id.strip(),
                        role="consultant",
                        api_key=config.get_model_api_key(model_id)
                    ))
        else:
            # Parse from YAML, inject API keys
            consultant_models = []
            for m in consultant_models_data:
                if "api_key" not in m or not m["api_key"]:
                    m["api_key"] = config.get_model_api_key(m.get("model_id", ""))
                consultant_models.append(ModelConfig(**m))
        
        # Parse verifier model
        verifier_data = data.get("verifier_model")
        if verifier_data:
            if "api_key" not in verifier_data or not verifier_data["api_key"]:
                verifier_data["api_key"] = config.get_model_api_key(verifier_data.get("model_id", ""))
            verifier_model = ModelConfig(**verifier_data)
        elif config.VERIFIER_MODEL:
            # Create from environment
            verifier_model = ModelConfig(
                name="verifier",
                model_id=config.VERIFIER_MODEL,
                role="verifier",
                api_key=config.get_model_api_key(config.VERIFIER_MODEL),
                temperature=config.VERIFIER_TEMPERATURE
            )
        else:
            verifier_model = None
        
        # Parse datasets
        datasets = [
            DatasetConfig(**d) for d in data.get("datasets", [])
        ]
        
        # Parse parallel config
        parallel_execution = [
            ParallelConfig(**p) for p in data.get("parallel_execution", [])
        ]
        
        # Parse verification
        verif_data = data.get("verification", {})
        verification = VerificationConfig(**verif_data) if verif_data else VerificationConfig()
        
        # Parse retrieval config
        retrieval_data = data.get("retrieval")
        retrieval = None
        if retrieval_data:
            # Inject env vars if not provided
            if "base_url" not in retrieval_data or not retrieval_data["base_url"]:
                retrieval_data["base_url"] = config.CHATNOIR_BASE_URL
            if "api_key" not in retrieval_data or not retrieval_data["api_key"]:
                retrieval_data["api_key"] = config.CHATNOIR_API_KEY
            if "corpus" not in retrieval_data or not retrieval_data["corpus"]:
                retrieval_data["corpus"] = config.CHATNOIR_DEFAULT_CORPUS
            
            retrieval = RetrievalConfig(**retrieval_data)
        
        return SimulationTemplate(
            id=data["id"],
            name=data["name"],
            mode=SimulationMode(data.get("mode", "standard")),
            teacher_models=teacher_models,
            consultant_models=consultant_models,
            verifier_model=verifier_model,
            workflows=data.get("workflows", []),
            datasets=datasets,
            retrieval=retrieval,
            max_iterations=data.get("max_iterations", config.MAX_ITERATIONS),
            similarity_metric=SimilarityMetric(data.get("similarity_metric", config.SIMILARITY_METRIC)),
            similarity_threshold=data.get("similarity_threshold", config.SIMILARITY_THRESHOLD),
            parallel_execution=parallel_execution,
            verification=verification,
            output_dir=data.get("output_dir", config.OUTPUT_DIR),
            mode_config=data.get("mode_config", {})
        )
    
    def validate(self, template: SimulationTemplate) -> Tuple[bool, List[str]]:
        """Validate simulation template"""
        errors = []
        
        # Validate workflows exist
        for wf_id in template.workflows:
            try:
                workflow = self.workflow_loader.load_workflow(wf_id)
                
                # Check last component is finalizer/synthesis
                if workflow.components:
                    last_comp = workflow.components[-1]
                    terminal_types = ["finalizer", "answer_drafter", "teacher_guided_agent_step"]
                    if last_comp.get("type") not in terminal_types:
                        errors.append(
                            f"Workflow '{wf_id}': Last component must be one of {terminal_types}, "
                            f"got '{last_comp.get('type')}'"
                        )
            except FileNotFoundError:
                errors.append(f"Workflow not found: {wf_id}")
        
        # Validate models
        if not template.teacher_models:
            errors.append("At least one teacher model required")
        
        # Validate datasets for standard/adaptive modes
        if not template.datasets and template.mode != SimulationMode.EXPLORATORY:
            errors.append("At least one dataset required for standard/adaptive modes")
        
        # Validate parallel execution references
        for parallel in template.parallel_execution:
            consultant_names = [m.name for m in template.consultant_models]
            for model_name in parallel.consultant_models:
                if model_name not in consultant_names:
                    errors.append(f"Unknown consultant model: {model_name}")
        
        return len(errors) == 0, errors

