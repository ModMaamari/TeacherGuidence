# Providers — using any model as teacher, judge, agent or student

Every model that is called over an API is addressed by a **provider-prefixed model id**.
The prefix selects the client; the rest is passed to the provider as the model name.
Credentials live in `.env` (copy `.env.example`; never commit it) or in the environment.

| Prefix | Talks to | Configuration | Example id |
|---|---|---|---|
| `oai/<model>` | any OpenAI-compatible `/v1/chat/completions` endpoint | `OAI_BASE_URL` (ends in `/v1`), `OAI_API_KEY` (optional) | `oai/deepseek-v4-flash` |
| `oai-<name>/<model>` | a second/third OpenAI-compatible endpoint | `OAI_<NAME>_BASE_URL`, `OAI_<NAME>_API_KEY` (NAME upper-cased, `-`→`_`) | `oai-judge/moonshotai/Kimi-K2.6` |
| `vllm/<served-name>` | the local vLLM student server | `VLLM_ENDPOINT` (default `http://127.0.0.1:8300`) | `vllm/student` |
| `edenchat/<provider>/<model>` | EdenAI, chat-completions route | `EDENAI_API_KEY` | `edenchat/flexai/DeepSeek-V4-Flash-0731` |
| `edenai/<provider>/<model>` | EdenAI, Responses route | `EDENAI_API_KEY` | `edenai/lilac/minimaxai/minimax-m3` |
| `nvidia/<model>` | NVIDIA NIM | `NVIDIA_API_KEY` | `nvidia/moonshotai/kimi-k2.6` |
| `custom/<model>` | OpenRouter-style endpoint (`<endpoint>/v1/chat/completions`, asks for per-call cost) | `CUSTOM_LLM_ENDPOINT`, `CUSTOM_LLM_API_KEY` | `custom/deepseek/deepseek-v4-flash` |
| `ollama/<model>` | local Ollama | `OLLAMA_ENDPOINT` | `ollama/qwen3.5:4b` |
| plain name | native SDKs by name pattern (`gpt-*`→OpenAI, `claude-*`→Anthropic, `gemini-*`→Google, `mistral-*`→Mistral) | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY` | `gpt-4o` |
| `mock/<anything>` | offline canned responses (tests only) | none | `mock/teacher` |

`oai` is the recommended route for a teacher or judge: it works with vLLM, TGI,
LiteLLM, llama.cpp, most university gateways and most commercial APIs, sends no vendor
extensions, and records `usage.cost` only when the server returns it. Reasoning models
that put their chain-of-thought in `reasoning_content` are supported (only `content`
is used; give them a large `--teacher-max-tokens`).

## Fallback chains

Wherever a model id is accepted, a comma-separated list is a fallback chain:

```
--teacher oai-teacher/deepseek-v4-flash,edenchat/flexai/DeepSeek-V4-Flash-0731
--judge  oai-judge/Kimi-K2.6,oai-judge/MiniMax-M3,oai-judge/Mistral-Medium
```

Each call tries the first model; on any error (rate limit, timeout, malformed reply) it
falls through to the next. Unconfigured providers are skipped up front. A provider that
times out three times in a row is bypassed for a cooldown and re-probed afterwards.
Every episode records which model actually answered (`teacher_models_used`;
`judge_model` on verdicts).

## Timeouts and retries

`OAI_TIMEOUT` (default 180 s) is a hard wall-clock cap per request; `EDENAI_TIMEOUT` and
`CUSTOM_TIMEOUT` likewise. Rate-limited calls back off exponentially (honouring
`Retry-After`) before falling through. `LLM_MAX_RETRIES` (default 3) sets the per-call
retry count for the last model in a chain.

## Which role uses which id

| Role | Flag | Typical id |
|---|---|---|
| student (local) | `--student vllm --served-model <name>` or `--student hf --model <hf id> [--adapter dir]` | — |
| student during collection | `collect_episodes.py --student` | `vllm/student` |
| teacher (collection, guided arm) | `--teacher` | `oai-teacher/<model>` |
| agent in the teacher-alone arm | `--agent-model` | `oai-teacher/<model>` |
| judge | `judge.py --judge` | `oai-judge/<model>` |

## Testing a provider

```bash
.venv/bin/python -c "
import asyncio, tgd
from agentsim.clients.llm_client import LLMClient
print(asyncio.run(LLMClient().get_completion('Reply with the single word OK.', model='oai-teacher/<model>', temperature=0, max_tokens=20)))"
```
