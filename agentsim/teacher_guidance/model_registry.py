"""The teachers and students the teacher-guidance corpus is generated with.

Configs reference a model by *name* ("kimi-k3", "qwen-0.8b") and this module resolves it to
a router chain or an HF id. That keeps provider routing -- which changes as credit runs out
or a gateway breaks -- out of every YAML and shell script, and gives the dataset card one
authoritative place to read licences and provenance from.

**Teachers** are addressed through an ordered fallback chain
(``LLMClient.get_completion_with_fallback``): each entry is tried in turn and the episode
records which one actually answered. Free self-hosted gateways come first so a run costs
nothing until it has to.

**Students** are served locally by vLLM under their real HF id, so every trace records the
model that produced it.

All models here are open-weight and all source datasets are permissively licensed, so the
resulting corpus is redistributable -- unlike agent-trace corpora distilled from
proprietary APIs. Verify :data:`TEACHERS` licences before publishing.

Availability last verified 2026-08-02 (see ``scripts/probe_models.py`` to re-check):

===================  ==========================================  ==============
teacher              source                                      status
===================  ==========================================  ==============
kimi-k3              edenai nebius/moonshotai/Kimi-K3             OK
kimi-k3              openrouter moonshotai/kimi-k3               no credit
glm-5.2              edenai lilac/zai-org/glm-5.2                 HTTP 500 (gateway bug)
glm-5.2              openrouter z-ai/glm-5.2                     no credit
deepseek-v4-flash    fau deepseek-ai/DeepSeek-V4-Flash            OK (free)
deepseek-v4-flash    openrouter deepseek/deepseek-v4-flash-0731  no credit
===================  ==========================================  ==============
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = [
    "STUDENTS",
    "TEACHERS",
    "ModelSource",
    "StudentSpec",
    "TeacherSpec",
    "combination_id",
    "student_names",
    "teacher_names",
    "get_student",
    "get_teacher",
]


@dataclass(frozen=True)
class ModelSource:
    """One provider that can serve a teacher, as an agentsim model id."""

    model_id: str          #: e.g. "edenai/nebius/moonshotai/Kimi-K3"
    provider: str          #: "edenai" | "openrouter" | "fau"
    free: bool = False
    note: str = ""


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    display_name: str
    license: str
    homepage: str
    #: Ordered fallback chain -- free/working sources first.
    sources: Sequence[ModelSource]
    #: Reasoning model? Affects the completion budget it needs, not the parsing (the
    #: parser strips inline chain-of-thought; see json_utils.strip_reasoning_blocks).
    reasoning: bool = True
    #: Generous defaults: a reasoning teacher spends completion budget thinking before
    #: it emits the JSON verdict, and a truncated verdict is an unusable episode.
    max_tokens: int = 2500
    max_tokens_retry: int = 4000

    @property
    def router(self) -> List[str]:
        return [s.model_id for s in self.sources]

    @property
    def primary(self) -> str:
        return self.sources[0].model_id


@dataclass(frozen=True)
class StudentSpec:
    name: str
    hf_id: str
    license: str
    params_b: float
    #: vLLM cannot LoRA-serve composite architectures (Qwen3.5's
    #: Qwen3_5ForConditionalGeneration wraps the LM as a tower), so an adapter must be
    #: merged into a full checkpoint before serving. See merge_qwen_composite.py.
    needs_merge_for_vllm: bool = False
    note: str = ""

    @property
    def served_model(self) -> str:
        """How a worker addresses this student once vLLM is serving it."""
        return f"vllm/{self.hf_id}"


# ---------------------------------------------------------------------------
# Teachers -- 3 models, each with a multi-source fallback chain
# ---------------------------------------------------------------------------
TEACHERS: Dict[str, TeacherSpec] = {
    "kimi-k3": TeacherSpec(
        name="kimi-k3",
        display_name="Kimi K3",
        license="Modified MIT (Kimi K3 License)",
        homepage="https://huggingface.co/moonshotai/Kimi-K3",
        sources=(
            ModelSource("edenai/nebius/moonshotai/Kimi-K3", "edenai",
                        note="verified working; billed per token"),
            ModelSource("custom/moonshotai/kimi-k3", "openrouter",
                        note="model exists; key limit currently exhausted"),
        ),
    ),
    "glm-5.2": TeacherSpec(
        name="glm-5.2",
        display_name="GLM-5.2",
        license="MIT",
        homepage="https://huggingface.co/zai-org/GLM-5.2",
        sources=(
            # OpenRouter is listed FIRST for this teacher: the EdenAI route is currently
            # broken gateway-side (deterministic HTTP 500, '_Parser' object has no
            # attribute 'extract_response_outputs'), so it cannot be the primary.
            ModelSource("custom/z-ai/glm-5.2", "openrouter",
                        note="model exists; key limit currently exhausted"),
            ModelSource("edenai/lilac/zai-org/glm-5.2", "edenai",
                        note="HTTP 500 from EdenAI as of 2026-08-02; kept as fallback"),
        ),
    ),
    "deepseek-v4-flash": TeacherSpec(
        name="deepseek-v4-flash",
        display_name="DeepSeek-V4-Flash",
        license="MIT",
        homepage="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
        sources=(
            # Self-hosted on the free academic gateway: no cost and no rate-limit spend,
            # so it leads the chain.
            ModelSource("fau/deepseek-ai/DeepSeek-V4-Flash", "fau", free=True,
                        note="verified working and free"),
            ModelSource("custom/deepseek/deepseek-v4-flash-0731", "openrouter",
                        note="model exists; key limit currently exhausted"),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Students -- 4 open-weight models, 0.8B to 3B, three families
# ---------------------------------------------------------------------------
STUDENTS: Dict[str, StudentSpec] = {
    "g9v3-3b": StudentSpec(
        name="g9v3-3b", hf_id="ai9stars/G9v3-3B", license="apache-2.0", params_b=3.0,
        note="LlamaForCausalLM; standard vLLM serving",
    ),
    "qwen-2b": StudentSpec(
        name="qwen-2b", hf_id="Qwen/Qwen3.5-2B", license="apache-2.0", params_b=2.0,
        needs_merge_for_vllm=True, note="composite Qwen3_5ForConditionalGeneration",
    ),
    "qwen-0.8b": StudentSpec(
        name="qwen-0.8b", hf_id="Qwen/Qwen3.5-0.8B", license="apache-2.0", params_b=0.8,
        needs_merge_for_vllm=True, note="composite Qwen3_5ForConditionalGeneration",
    ),
    "granite-3b": StudentSpec(
        name="granite-3b", hf_id="ibm-granite/granite-4.1-3b", license="apache-2.0",
        params_b=3.0, note="GraniteForCausalLM; serves LoRA directly",
    ),
}


def teacher_names() -> List[str]:
    return list(TEACHERS)


def student_names() -> List[str]:
    return list(STUDENTS)


def get_teacher(name: str) -> TeacherSpec:
    try:
        return TEACHERS[name]
    except KeyError:
        raise KeyError(f"unknown teacher {name!r}; known: {teacher_names()}") from None


def get_student(name: str) -> StudentSpec:
    try:
        return STUDENTS[name]
    except KeyError:
        raise KeyError(f"unknown student {name!r}; known: {student_names()}") from None


def combination_id(teacher: str, student: str) -> str:
    """Stable id for one teacher-student cell of the collection matrix."""
    return f"{teacher}__{student}"
