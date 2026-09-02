"""
AgentSim CLI - Command-line interface for workflow execution.
"""

import asyncio
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agentsim.workflow.loader import WorkflowLoader
from agentsim.workflow.executor import WorkflowExecutor
from agentsim.components.base import ComponentRegistry, ComponentCategory


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='agentsim',
        description='AgentSim - Modular agentic simulation framework'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List workflows
    subparsers.add_parser('list', help='List available workflows')
    
    # Run workflow
    run_parser = subparsers.add_parser('run', help='Execute a workflow')
    run_parser.add_argument('workflow', help='Workflow ID')
    run_parser.add_argument('--query', '-q', required=True, help='Query to execute')
    run_parser.add_argument('--output', '-o', help='Output file (JSON)')
    
    # Validate workflow
    validate_parser = subparsers.add_parser('validate', help='Validate workflow')
    validate_parser.add_argument('file', help='Workflow YAML file')
    
    # List components
    comp_parser = subparsers.add_parser('components', help='List components')
    comp_parser.add_argument('--category', '-c', help='Filter by category')
    
    # Show info
    subparsers.add_parser('info', help='Show system information')
    
    # Simulation commands
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('template', help='Simulation template ID')
    sim_parser.add_argument('--validate-only', action='store_true', help='Only validate')

    # Discover models
    discover_parser = subparsers.add_parser('discover', help='Discover available models from endpoints')
    discover_parser.add_argument('--custom', action='store_true', help='Discover custom endpoint models')
    discover_parser.add_argument('--ollama', action='store_true', help='Discover Ollama models')
    discover_parser.add_argument('--all', action='store_true', help='Discover all configured endpoints')
    discover_parser.add_argument('--endpoint', help='Custom endpoint URL')
    discover_parser.add_argument('--api-key', help='API key for custom endpoint')
    
    # Seed selection
    seed_parser = subparsers.add_parser('seed-select', help='Coverage-driven seed selection')
    seed_parser.add_argument('--data-dir', type=str, default=str(Path(__file__).parent.parent / "data" / "datasets"),
                             help='Datasets base directory')
    seed_parser.add_argument('--dataset', type=str, required=True,
                             help='Dataset name (e.g., msmarco, quasar_t, trec_tot)')
    seed_parser.add_argument('--split', type=str, required=True, choices=['train', 'dev', 'test', 'eval', 'validation'],
                             help='Dataset split')
    seed_parser.add_argument('--retrieval', type=str, default='opensearch', choices=['opensearch', 'chatnoir'],
                             help='Retrieval backend (default: opensearch)')
    seed_parser.add_argument('--index', type=str, default='msmarco-v2.1-segmented',
                             help='Index/corpus name for retrieval (default: msmarco-v2.1-segmented)')
    seed_parser.add_argument('--topk', type=int, default=20, help='Top-k documents per candidate (default: 20)')
    seed_parser.add_argument('--max-candidates', type=int, default=50000, help='Max candidate queries to consider')
    seed_parser.add_argument('--num-seeds', type=int, default=1000, help='Number of seeds to select')
    seed_parser.add_argument('--clusters', type=int, default=100, help='Number of clusters for quotas (default: 100)')
    seed_parser.add_argument('--novelty-threshold', type=float, default=0.6,
                             help='Reject queries with overlap fraction above this (default: 0.6)')
    seed_parser.add_argument('--lambda-mmr', type=float, default=0.7,
                             help='MMR tradeoff lambda (default: 0.7)')
    seed_parser.add_argument('--output', type=str, default='seeds.jsonl', help='Output seeds file')
    seed_parser.add_argument('--opensearch-fields', type=str, default='segment^3,segment,title,body',
                             help='Comma-separated fields for OpenSearch multi_match (default: segment^3,segment,title,body)')
    seed_parser.add_argument('--request-delay', type=float, default=0.0,
                             help='Seconds to sleep between retrieval calls (recommended >=0.25 for ChatNoir)')
    seed_parser.add_argument('--prior-docs', type=str, help='Optional JSONL file of prior covered doc_ids')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'list':
        cmd_list()
    elif args.command == 'run':
        asyncio.run(cmd_run(args.workflow, args.query, args.output))
    elif args.command == 'validate':
        cmd_validate(args.file)
    elif args.command == 'components':
        cmd_components(args.category)
    elif args.command == 'info':
        cmd_info()
    elif args.command == 'simulate':
        asyncio.run(cmd_simulate(args.template, args.validate_only))
    elif args.command == 'discover':
        asyncio.run(cmd_discover(args))
    elif args.command == 'seed-select':
        asyncio.run(cmd_seed_select(args))


def cmd_list():
    """List available workflows"""
    loader = WorkflowLoader()
    workflows = loader.list_workflows_with_info()
    
    print("\nAvailable Workflows:")
    print("=" * 60)
    for wf in workflows:
        print(f"\n{wf['name']} ({wf['id']})")
        print(f"  {wf['description']}")
        print(f"  Components: {wf['component_count']}")


async def cmd_run(workflow_id: str, query: str, output: Optional[str] = None):
    """Execute a workflow"""
    print(f"\nExecuting workflow: {workflow_id}")
    print(f"Query: {query}\n")
    
    try:
        loader = WorkflowLoader()
        workflow = loader.load_workflow(workflow_id)
        
        # Note: This requires clients to be configured
        # In real usage, you'd pass actual clients here
        print("⚠️  Note: Running without external clients (LLM, OpenSearch)")
        print("   For full execution, use the API or provide clients\n")
        
        executor = WorkflowExecutor()
        context = await executor.execute(workflow, query)
        
        result = {
            "success": True,
            "query": query,
            "workflow": workflow_id,
            "evidence_count": len(context.evidence_store),
            "steps": len(context.messages),
            "final_answer": context.metadata.get("final_answer"),
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Results saved to {output}")
        else:
            print("Results:")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_validate(filepath: str):
    """Validate a workflow file"""
    try:
        import yaml
        
        with open(filepath) as f:
            data = yaml.safe_load(f)
        
        loader = WorkflowLoader()
        workflow = loader.load_workflow_from_dict(data)
        is_valid, errors = loader.validate_workflow(workflow)
        
        if is_valid:
            print(f"✓ Workflow '{workflow.name}' is valid")
            print(f"  Components: {len(workflow.components)}")
        else:
            print(f"✗ Workflow validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_components(category: Optional[str] = None):
    """List available components"""
    if category:
        try:
            cat = ComponentCategory(category)
            components = ComponentRegistry.list_components(cat)
            print(f"\n{category.upper()} Components:")
        except ValueError:
            print(f"Invalid category: {category}")
            print(f"Valid: {[c.value for c in ComponentCategory]}")
            sys.exit(1)
    else:
        components = ComponentRegistry.list_components()
        print("\nAll Components:")
    
    print("=" * 60)
    for comp in sorted(components):
        try:
            spec = ComponentRegistry.get_spec(comp)
            print(f"\n{comp}")
            print(f"  Category: {spec.category.value}")
            print(f"  {spec.description}")
        except:
            print(f"\n{comp}")


def cmd_info():
    """Show system information"""
    from agentsim import __version__
    
    all_components = ComponentRegistry.list_components()
    components_by_cat = {}
    for cat in ComponentCategory:
        components_by_cat[cat.value] = len(ComponentRegistry.list_components(cat))
    
    loader = WorkflowLoader()
    workflows = loader.list_available_workflows()
    
    print("\nAgentSim Information")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Total Components: {len(all_components)}")
    print(f"Total Workflows: {len(workflows)}")
    print("\nComponents by Category:")
    for cat, count in components_by_cat.items():
        print(f"  {cat}: {count}")
    print("\nWorkflows:")
    for wf in workflows:
        print(f"  - {wf}")


async def cmd_simulate(template_id: str, validate_only: bool = False):
    """Run simulation"""
    import json
    import uuid
    from agentsim.simulation.loader import SimulationLoader
    from agentsim.workflow.loader import WorkflowLoader
    from agentsim.simulation.modes import StandardRunner, AdaptiveRunner, ExploratoryRunner
    from agentsim.simulation.schema import SimulationMode
    from agentsim.utils.trace_exporter import TraceExporter
    
    try:
        loader = SimulationLoader()
        template = loader.load(template_id)
        
        print(f"\nSimulation: {template.name}")
        print(f"Mode: {template.mode.value}")
        print(f"Teacher models: {len(template.teacher_models)}")
        print(f"Workflows: {', '.join(template.workflows)}")
        print(f"Datasets: {len(template.datasets)}")
        
        # Validate
        is_valid, errors = loader.validate(template)
        
        if not is_valid:
            print("\n✗ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print("✓ Template valid")
        
        if validate_only:
            return
        
        # Execute simulation
        print(f"\n▶ Starting {template.mode.value} simulation...")
        
        # Initialize clients from config
        clients = _initialize_clients(template)
        
        # Load workflow
        workflow_loader = WorkflowLoader()
        workflow = workflow_loader.load_workflow(template.workflows[0])
        
        # Load dataset
        dataset = _load_dataset(template.datasets[0])
        
        # Check for checkpoint to resume
        # Ensure the output directory exists before writing the checkpoint, so
        # templates with a nested output_dir do not fail on first run.
        Path(template.output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(template.output_dir) / f"checkpoint_{template.id}.json"
        completed_samples = set()
        dataset_name = template.datasets[0].name
        resume = False
        state = {}
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    existing_state = json.load(f)
            except Exception:
                existing_state = None
            
            if existing_state and existing_state.get("status") == "running":
                run_uuid = existing_state.get("run_uuid")
                if run_uuid:
                    tentative_run_dir = Path(template.output_dir) / run_uuid
                    if tentative_run_dir.exists():
                        resume = True
                        state = existing_state
                        dataset_name = existing_state.get("dataset_name", dataset_name)
                        completed_samples = set(existing_state.get("completed_samples", []))
                        run_dir = tentative_run_dir
                        print(f"Resuming simulation run: {run_uuid}")
        
        if not resume:
            run_uuid = str(uuid.uuid4())[:8]
            run_dir = Path(template.output_dir) / run_uuid
            state = {
                "template_id": template.id,
                "status": "running",
                "run_uuid": run_uuid,
                "dataset_name": dataset_name,
                "completed_samples": [],
                "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"Simulation run: {run_uuid}")
        else:
            print(f"Simulation run (resume): {run_uuid}")
        
        # Ensure directories exist
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = run_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Dataset: {dataset_name}")
        
        if not dataset:
            print("\n✗ Dataset is empty after applying filters/num_samples.")
            sys.exit(1)
        
        total_seeds = len(dataset)
        if template.mode == SimulationMode.EXPLORATORY:
            max_expl = template.mode_config.get('max_explorations', 10)
            print(f"Exploratory mode: Processing {total_seeds} seeds; each seed can expand up to {max_expl} queries\n")
        else:
            print(f"Processing {total_seeds} samples...\n")
        
        all_results = []
        for i, sample in enumerate(dataset):
            sample_id = f"sample_{i+1:03d}"
            sample_query = sample.get('query', sample.get('question', 'N/A'))
            print(f"Sample {i+1}/{len(dataset)} [{sample_id}]: {sample_query[:60]}...")
            
            # Create sample subdirectory
            sample_dir = dataset_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Skip if already completed (checkpoint or success marker)
            completion_marker = sample_dir / "_SUCCESS"
            if sample_id in completed_samples or completion_marker.exists():
                print(f"  ↷ Skipping (already completed).")
                completed_samples.add(sample_id)
                state["completed_samples"] = sorted(completed_samples)
                with open(checkpoint_path, 'w') as f:
                    json.dump(state, f, indent=2)
                continue
            
            try:
                # Create trace exporter that writes on-the-go
                from agentsim.utils.trace_exporter import TraceExporter
                exporter = TraceExporter(output_dir=str(sample_dir))
                
                # Create workflow executor with trace exporter
                from agentsim.workflow.executor import WorkflowExecutor
                workflow_executor = WorkflowExecutor(
                    **clients,
                    retrieval_config=template.retrieval,
                    trace_exporter=exporter
                )
                
                # Create runner based on mode
                if template.mode == SimulationMode.STANDARD:
                    runner = StandardRunner(template, clients, workflow_executor)
                elif template.mode == SimulationMode.ADAPTIVE:
                    runner = AdaptiveRunner(template, clients, workflow_executor)
                elif template.mode == SimulationMode.EXPLORATORY:
                    runner = ExploratoryRunner(template, clients, workflow_executor, run_uuid=run_uuid)
                
                result = await runner.run(workflow, sample, sample_id=sample_id)
                all_results.append(result)
                
                # Export summary files (traces were streamed during execution)
                query = sample.get("query") or sample.get("question", "")
                expected_answer = sample.get("answer", "")
                exporter.export_summary_files(
                    template=template,
                    query=query,
                    expected_answer=expected_answer,
                    workflow=workflow,
                    context=result.get("context"),
                    result=result,
                    run_id=f"{run_uuid}_{dataset_name}_{sample_id}"
                )

                # Teacher Guidance: export clean episode-level files when the run
                # produced teacher-guided steps.
                tg_context = result.get("context")
                if tg_context is not None and "teacher_guided_steps" in tg_context.metadata:
                    from agentsim.teacher_guidance.episode_exporter import (
                        TeacherGuidanceEpisodeExporter,
                    )
                    TeacherGuidanceEpisodeExporter().export_episode(tg_context, str(sample_dir))

                print(f"  ✓ Exported")
                
                # Mark sample as completed (checkpoint)
                completion_payload = {
                    "seed_id": sample_id,
                    "completed_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                }
                with open(completion_marker, 'w') as f:
                    json.dump(completion_payload, f, indent=2)
                
                completed_samples.add(sample_id)
                state["completed_samples"] = sorted(completed_samples)
                with open(checkpoint_path, 'w') as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                import traceback
                print(f"  ✗ Error: {e}")
                print(traceback.format_exc())
                all_results.append({"error": str(e), "sample": sample})
        
        successful_in_run = sum(1 for r in all_results if 'error' not in r)
        print(f"\n✓ Simulation complete")
        print(f"Output: {run_dir}")
        if all_results:
            print(f"  └─ This run: {successful_in_run}/{len(all_results)} seeds processed")
        print(f"  └─ Overall completed: {len(completed_samples)}/{total_seeds} seeds")
        print(f"Run UUID: {run_uuid}")
        
        state["status"] = "completed"
        state["completed_at"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        state["completed_samples"] = sorted(completed_samples)
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        
    except Exception as e:
        import traceback
        print(f"\n✗ Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


def _load_dataset(dataset_config):
    """Load dataset from file"""
    import json
    from pathlib import Path
    
    path = Path(dataset_config.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    dataset = []
    with open(path) as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    # Apply num_samples limit
    if dataset_config.num_samples > 0:
        dataset = dataset[:dataset_config.num_samples]
    
    return dataset


def _initialize_clients(template):
    """Initialize LLM and retrieval clients from config"""
    from agentsim.clients import LLMClient
    from agentsim.config import config
    
    # Initialize LLM client with default model from template
    default_model = template.teacher_models[0].model_id if template.teacher_models else None
    llm_client = LLMClient(default_model=default_model)
    print(f"✓ LLM client initialized (default: {llm_client.default_model})")
    
    # Check which models are configured
    teacher_models = [m.model_id for m in template.teacher_models]
    print(f"  Teacher models: {', '.join(teacher_models)}")
    
    # Verify API keys for required providers
    missing_keys = []
    for model in teacher_models:
        provider = config.get_provider_from_model_id(model)
        api_key = config.get_api_key_for_provider(provider)
        # Locally-served providers need no API key.
        if provider not in ["ollama", "vllm"] and not api_key:
            missing_keys.append(f"{provider.upper()}_API_KEY")
    
    if missing_keys:
        print(f"⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
        print("   Set these in .env to enable model calls")
    
    # ChatNoir client
    chatnoir_client = None
    if config.CHATNOIR_ENABLED and config.CHATNOIR_API_KEY:
        print(f"✓ ChatNoir configured")
        # ChatNoir client would be initialized here
        # For now, using httpx directly in the component
    elif config.CHATNOIR_ENABLED:
        print(f"⚠️  ChatNoir enabled but no API key set")
    
    return {
        "llm_client": llm_client,
        "chatnoir_client": chatnoir_client,
        "opensearch_client": None,
        "vector_client": None
    }

async def cmd_seed_select(args):
    """Run coverage-driven seed selection."""
    from agentsim.tools.seed_selector import SeedSelectorConfig, run_seed_selection
    cfg = SeedSelectorConfig(
        data_dir=Path(args.data_dir),
        dataset_name=args.dataset,
        split=args.split,
        retrieval=args.retrieval,
        index_name=args.index,
        top_k=args.topk,
        max_candidates=args.max_candidates,
        num_seeds=args.num_seeds,
        clusters=args.clusters,
        novelty_threshold=args.novelty_threshold,
        lambda_mmr=args.lambda_mmr,
        output_path=Path(args.output),
        opensearch_fields=tuple([s.strip() for s in args.opensearch_fields.split(",") if s.strip()]),
        prior_docs_file=Path(args.prior_docs) if args.prior_docs else None,
        request_delay=args.request_delay
    )
    await run_seed_selection(cfg)

async def cmd_discover(args):
    """Discover available models from endpoints"""
    from agentsim.utils.endpoint_discovery import (
        discover_custom_endpoint_models,
        discover_ollama_models,
        format_models_table
    )
    from agentsim.config import config
    
    try:
        if args.all or args.custom:
            endpoint = args.endpoint or config.CUSTOM_LLM_ENDPOINT
            api_key = args.api_key or config.CUSTOM_LLM_API_KEY
            
            if endpoint:
                print(f"\nDiscovering models from: {endpoint}")
                models = await discover_custom_endpoint_models(endpoint, api_key)
                
                if models:
                    print(format_models_table(models, "Custom Endpoint"))
                    print("\nUsage in YAML:")
                    print("  consultant_models:")
                    for model in models[:3]:  # Show first 3 examples
                        model_id = model.get('id', model.get('name'))
                        print(f"    - model_id: custom/{model_id}")
                else:
                    print("✗ No models found or endpoint unreachable")
            else:
                print("✗ No custom endpoint configured")
                print("  Set CUSTOM_LLM_ENDPOINT in .env")
        
        if args.all or args.ollama:
            endpoint = config.OLLAMA_ENDPOINT
            
            if config.OLLAMA_ENABLED:
                print(f"\nDiscovering models from Ollama: {endpoint}")
                models = await discover_ollama_models(endpoint)
                
                if models:
                    print(format_models_table(models, "Ollama"))
                    print("\nUsage in YAML:")
                    print("  teacher_models:")
                    for model in models[:3]:  # Show first 3 examples
                        model_id = model.get('id', model.get('name'))
                        print(f"    - model_id: ollama/{model_id}")
                else:
                    print("✗ No Ollama models found or Ollama not running")
                    print(f"  Make sure Ollama is running at {endpoint}")
            else:
                print("✗ Ollama not enabled")
                print("  Set OLLAMA_ENABLED=true in .env")
        
        if not (args.all or args.custom or args.ollama):
            print("Specify --custom, --ollama, or --all")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

