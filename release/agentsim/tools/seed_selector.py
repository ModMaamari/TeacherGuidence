"""
Seed selection utility for coverage-driven exploration.

Features:
- Loads candidate queries from dataset jsonl files (train/dev/test).
- Retrieves top-k doc_ids per candidate using a chosen backend (opensearch default, chatnoir optional).
- Selects a seed batch maximizing marginal novel documents with MMR diversity and simple cluster quotas.
- Writes seeds.jsonl with selected queries.
"""

from __future__ import annotations

import asyncio
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Set, Optional

import httpx
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from urllib.parse import urlparse

from agentsim.config import config
from agentsim.clients.llm_client import LLMClient


SUPPORTED_SPLITS = {"train", "dev", "test", "eval", "validation"}


@dataclass
class SeedSelectorConfig:
    data_dir: Path
    dataset_name: str
    split: str
    retrieval: str = "opensearch"  # opensearch | chatnoir
    index_name: str = "msmarco-v2.1-segmented"
    top_k: int = 20
    max_candidates: int = 50000
    num_seeds: int = 1000
    clusters: int = 100
    novelty_threshold: float = 0.6
    lambda_mmr: float = 0.7
    output_path: Path = Path("./seeds.jsonl")
    opensearch_fields: Tuple[str, ...] = ("segment^3", "segment", "title", "body")
    prior_docs_file: Optional[Path] = None
    request_delay: float = 0.0  # seconds between retrieval calls (used for rate limiting)


def load_candidates(data_dir: Path, dataset_name: str, split: str, max_candidates: int) -> List[str]:
    """
    Load candidate queries from dataset directory.
    Expects jsonl files; tries to read any of these keys: 'query', 'question'.
    Splits are detected by filename containing the split string.
    """
    ds_dir = data_dir / dataset_name
    if not ds_dir.exists():
        # fallback to flat layout
        ds_dir = data_dir
    # Try both .jsonl and .json files, including versioned MSMARCO files
    files = list(ds_dir.glob("**/*.jsonl")) + list(ds_dir.glob("**/*.json"))
    candidates: List[str] = []
    for fp in files:
        name_lower = fp.name.lower()
        if split not in name_lower:
            continue
        
        logger.info(f"Reading {fp}")
        
        # Handle .jsonl (one JSON per line)
        if fp.suffix == ".jsonl":
            with fp.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except:
                        continue
                    q = obj.get("query") or obj.get("question") or obj.get("text")
                    if isinstance(q, str) and q.strip():
                        candidates.append(q.strip())
                        if len(candidates) >= max_candidates:
                            break
        
        # Handle .json (complete JSON object)
        elif fp.suffix == ".json":
            with fp.open() as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Treat as JSONL despite .json extension (e.g., Quasar-T)
                    logger.info(f"{fp} appears to be JSONL despite .json extension; falling back to line-by-line parsing")
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        q = obj.get("query") or obj.get("question") or obj.get("text")
                        if isinstance(q, str) and q.strip():
                            candidates.append(q.strip())
                            if len(candidates) >= max_candidates:
                                break
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load {fp}: {e}")
                    continue
                
                if isinstance(data, dict):
                    # Handle MSMARCO QA JSON structure:
                    # {
                    #   "query": {query_id: "query text", ...},
                    #   "answers": {...},
                    #   ...
                    # }
                    if "query" in data and isinstance(data["query"], dict):
                        logger.info(f"Detected MSMARCO QA dictionary with {len(data['query'])} query entries")
                        for q in list(data["query"].values())[:max_candidates]:
                            if isinstance(q, str) and q.strip():
                                candidates.append(q.strip())
                                if len(candidates) >= max_candidates:
                                    break
                    else:
                        # Generic dict: {query_id: query_text}
                        logger.info(f"Detected dictionary with {len(data)} entries; treating keys as query texts")
                        for query_id in list(data.keys())[:max_candidates]:
                            if isinstance(query_id, str) and query_id.strip():
                                candidates.append(query_id.strip())
                                if len(candidates) >= max_candidates:
                                    break
                # List of objects
                elif isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            q = obj.get("query") or obj.get("question") or obj.get("text")
                            if isinstance(q, str) and q.strip():
                                candidates.append(q.strip())
                                if len(candidates) >= max_candidates:
                                    break
        
        if len(candidates) >= max_candidates:
            break
    logger.info(f"Loaded {len(candidates)} candidates from split '{split}'")
    return candidates


async def opensearch_topk(client: httpx.AsyncClient, index: str, query: str, k: int, fields: Tuple[str, ...]) -> List[str]:
    """
    Query OpenSearch/Elasticsearch for top-k documents using multi_match.
    Returns document IDs (_id).
    """
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": list(fields),
                "type": "best_fields"
            }
        }
    }
    body["_source"] = {"excludes": ["emb"]}
    resp = await client.post(f"/{index}/_search", json=body)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])
    return [h.get("_id") or h.get("_source", {}).get("id", "") for h in hits if h]


async def chatnoir_topk(client: httpx.AsyncClient, query: str, k: int, corpus: str) -> List[str]:
    params = {"query": query, "index": corpus, "size": k}
    if config.CHATNOIR_API_KEY:
        params["apikey"] = config.CHATNOIR_API_KEY
    resp = await client.get(f"{config.CHATNOIR_BASE_URL}/_search", params=params)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for hit in data.get("results", []):
        doc_id = hit.get("trec_id", hit.get("uuid", ""))
        if doc_id:
            results.append(doc_id)
    return results[:k]


async def build_topk_map(queries: List[str], cfg: SeedSelectorConfig) -> Dict[str, List[str]]:
    """
    Build a mapping from query -> topk doc_ids via chosen backend.
    """
    topk_map: Dict[str, List[str]] = {}
    timeout = httpx.Timeout(60.0)
    if cfg.retrieval == "opensearch":
        scheme = "https" if config.OPENSEARCH_USE_SSL else "http"
        host_value = config.OPENSEARCH_HOST
        if host_value.startswith(("http://", "https://")):
            parsed = urlparse(host_value)
            scheme = parsed.scheme or scheme
            netloc = parsed.netloc or host_value
            base_url = f"{scheme}://{netloc}"
            if parsed.path and parsed.path != "/":
                base_url += parsed.path.rstrip("/")
        else:
            host = host_value.rstrip("/")
            port = config.OPENSEARCH_PORT
            if port == 443 and scheme == "https":
                base_url = f"{scheme}://{host}"
            elif port == 80 and scheme == "http":
                base_url = f"{scheme}://{host}"
            else:
                base_url = f"{scheme}://{host}:{port}"
        verify = config.OPENSEARCH_USE_SSL if scheme == "https" else False
        logger.debug(f"OpenSearch base_url resolved to {base_url}")
        auth = None
        if config.OPENSEARCH_USER and config.OPENSEARCH_PASSWORD:
            auth = (config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout, auth=auth, verify=verify) as client:
            for i, q in enumerate(queries):
                try:
                    topk_map[q] = await opensearch_topk(client, cfg.index_name, q, cfg.top_k, cfg.opensearch_fields)
                    if not topk_map[q]:
                        logger.debug(f"OpenSearch returned 0 hits for query {i}; query text='{q[:80]}'")
                except Exception as e:
                    logger.debug(f"OpenSearch error for query {i}: {repr(e)}", exc_info=True)
                    topk_map[q] = []
                if (i + 1) % 1000 == 0:
                    logger.info(f"Top-k fetched for {i+1}/{len(queries)} queries")
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i, q in enumerate(queries):
                try:
                    docs = await chatnoir_topk(client, q, cfg.top_k, cfg.index_name)
                    topk_map[q] = docs
                    if not docs:
                        logger.debug(f"ChatNoir returned 0 hits for query {i}; query text='{q[:80]}'")
                except Exception as e:
                    logger.debug(f"ChatNoir error for query {i}: {e}")
                    topk_map[q] = []
                if cfg.request_delay > 0:
                    await asyncio.sleep(cfg.request_delay)
                if (i + 1) % 1000 == 0:
                    logger.info(f"Top-k fetched for {i+1}/{len(queries)} queries")
    return topk_map


async def embed_queries(queries: List[str]) -> np.ndarray:
    """
    Build normalized embeddings for queries using local SentenceTransformer via LLMClient.
    """
    llm = LLMClient()
    # batch encode to avoid OOM
    embs: List[np.ndarray] = []
    batch_size = 256
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        # use get_embedding one by one to utilize async loop offloading inside client
        vecs = []
        for q in batch:
            v = await llm.get_embedding(q)
            vecs.append(np.array(v, dtype=np.float32))
        embs.append(np.vstack(vecs))
    X = np.vstack(embs) if embs else np.zeros((0, 384), dtype=np.float32)
    # normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms


def mmr_score(q_vec: np.ndarray, selected: List[np.ndarray], marginal: float, lambda_mmr: float) -> float:
    if not selected:
        return marginal
    sims = [float(q_vec @ s) for s in selected]
    return lambda_mmr * marginal - (1.0 - lambda_mmr) * max(sims)


def select_seeds(
    queries: List[str],
    X: np.ndarray,
    topk_map: Dict[str, List[str]],
    prior_docs: Set[str],
    cfg: SeedSelectorConfig
) -> List[str]:
    """
    Greedy selection by cluster quotas + MMR novelty.
    """
    n = len(queries)
    if n == 0:
        return []
    # clustering
    k = min(cfg.clusters, n)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(X)
    cluster_to_indices: Dict[int, List[int]] = {}
    for idx, c in enumerate(cluster_ids):
        cluster_to_indices.setdefault(int(c), []).append(idx)
    # selection
    S: Set[str] = set(prior_docs)
    selected_indices: List[int] = []
    selected_vecs: List[np.ndarray] = []
    cluster_cycle = list(cluster_to_indices.keys())
    cycle_ptr = 0
    while len(selected_indices) < cfg.num_seeds and cluster_cycle:
        c = cluster_cycle[cycle_ptr % len(cluster_cycle)]
        pool = [i for i in cluster_to_indices.get(c, []) if i not in selected_indices]
        best_i, best_score = None, -1e9
        for i in pool:
            docs = topk_map.get(queries[i], [])
            if not docs:
                continue
            overlap = len(set(docs) & S) / max(len(docs), 1)
            if overlap > cfg.novelty_threshold:
                continue
            marginal = len(set(docs) - S)
            score = mmr_score(X[i], selected_vecs, float(marginal), cfg.lambda_mmr)
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            # remove exhausted cluster
            cluster_cycle.remove(c)
            continue
        selected_indices.append(best_i)
        selected_vecs.append(X[best_i])
        S |= set(topk_map.get(queries[best_i], []))
        cycle_ptr += 1
    return [queries[i] for i in selected_indices]


def load_prior_docs(path: Optional[Path]) -> Set[str]:
    if not path:
        return set()
    try:
        docs: Set[str] = set()
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue
                # supports a line with {"doc_id": "..."} or {"documents": {"<id>": {...}}}
                d = obj.get("doc_id")
                if d:
                    docs.add(d)
        return docs
    except Exception:
        return set()


async def run_seed_selection(cfg: SeedSelectorConfig):
    # 1) load candidates
    queries = load_candidates(cfg.data_dir, cfg.dataset_name, cfg.split, cfg.max_candidates)
    if not queries:
        logger.error("No candidates loaded. Check data_dir/dataset_name/split.")
        return
    if cfg.retrieval == "chatnoir" and cfg.request_delay <= 0:
        cfg.request_delay = 0.25
        logger.info(f"Applying default ChatNoir request delay of {cfg.request_delay:.2f}s to avoid rate limits")
    # 2) topk map
    topk_map = await build_topk_map(queries, cfg)
    # 3) embeddings
    X = await embed_queries(queries)
    # 4) prior docs
    prior_docs = load_prior_docs(cfg.prior_docs_file)
    # 5) selection
    seeds = select_seeds(queries, X, topk_map, prior_docs, cfg)
    # 6) write seeds
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w") as f:
        for q in seeds:
            f.write(json.dumps({"query": q, "dataset": cfg.dataset_name, "split": cfg.split}) + "\n")
    logger.info(f"Wrote {len(seeds)} seeds to {cfg.output_path}")


