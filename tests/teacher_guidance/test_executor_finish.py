"""Verify the workflow executor stops early when a component returns FINISH."""

import asyncio

from agentsim.components.base import (
    BaseComponent,
    ComponentSpec,
    ComponentResult,
    ComponentCategory,
    ComponentRegistry,
)
from agentsim.workflow.loader import WorkflowDefinition
from agentsim.workflow.executor import WorkflowExecutor


@ComponentRegistry.register("_tg_test_marker")
class _MarkerComponent(BaseComponent):
    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="_tg_test_marker",
            category=ComponentCategory.CONTROL,
            description="test marker",
            config_schema={"verdict": {"type": "string", "default": "PROCEED"}, "tag": {"type": "string", "default": ""}},
        )

    async def execute(self, context):
        context.metadata.setdefault("ran", []).append(self.config.get("tag"))
        return ComponentResult(success=True, data={"verdict": self.config.get("verdict", "PROCEED")})


def _workflow():
    return WorkflowDefinition(
        id="wf", name="wf", description="", reasoning_style="",
        components=[
            {"type": "_tg_test_marker", "config": {"tag": "a", "verdict": "PROCEED"}},
            {"type": "_tg_test_marker", "config": {"tag": "b", "verdict": "FINISH"}},
            {"type": "_tg_test_marker", "config": {"tag": "c", "verdict": "PROCEED"}},
        ],
        config={"max_steps": 3},
    )


def test_executor_stops_on_finish():
    executor = WorkflowExecutor()
    context = asyncio.run(executor.execute(_workflow(), query="q"))
    # The third component must NOT run because the second returned FINISH.
    assert context.metadata["ran"] == ["a", "b"]
    assert context.messages[-1].stop_condition == "FINISH"
