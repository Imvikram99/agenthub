"""
Tiered Memory Architecture for Hub Gateway.

Layer 1: User Facts (Redis) — structured key-value pairs per user
         e.g., portfolio_path, last_analysis_date
         Written by: tool output fact hooks
         Read: at conversation start, injected into system prompt

Layer 2: User Preferences (Qdrant) — semantic/soft memories
         e.g., "prefers conservative investments", "don't touch payments"
         Written by: async LLM extraction (post-response, fire-and-forget)
         Read: at conversation start via similarity search
"""

import asyncio
import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as aioredis

logger = logging.getLogger("hub.memory")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
FACTS_PREFIX = "hub:facts"

_redis_client: Optional[aioredis.Redis] = None


async def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# ── Layer 1: User Facts (Redis) ─────────────────────────────────────────────


async def save_user_fact(user_id: str, key: str, value: str):
    """Save a structured fact for a user."""
    try:
        r = await _get_redis()
        await r.hset(f"{FACTS_PREFIX}:{user_id}", key, value)
        logger.info(f"Saved fact for user {user_id}: {key}={value[:100]}")
    except Exception as e:
        logger.error(f"Failed to save user fact: {e}")


async def get_user_facts(user_id: str) -> Dict[str, str]:
    """Get all structured facts for a user."""
    try:
        r = await _get_redis()
        facts = await r.hgetall(f"{FACTS_PREFIX}:{user_id}")
        return facts or {}
    except Exception as e:
        logger.error(f"Failed to get user facts: {e}")
        return {}


async def delete_user_fact(user_id: str, key: str):
    """Delete a specific fact for a user."""
    try:
        r = await _get_redis()
        await r.hdel(f"{FACTS_PREFIX}:{user_id}", key)
    except Exception as e:
        logger.error(f"Failed to delete user fact: {e}")


# ── Layer 2: User Preferences (Qdrant — async LLM extraction) ───────────────

# These are initialized lazily by gateway.py since they depend on OpenAI/Qdrant setup
_vector_store = None
_llm_client = None
_llm_model = None


def init_layer2(vector_store, llm_client, llm_model: str):
    """Initialize Layer 2 with shared instances from gateway."""
    global _vector_store, _llm_client, _llm_model
    _vector_store = vector_store
    _llm_client = llm_client
    _llm_model = llm_model


def recall_user_preferences(user_id: str, query: str) -> str:
    """Search Qdrant for relevant past preferences/context for this user."""
    if not _vector_store:
        return ""
    try:
        docs = _vector_store.similarity_search(
            query=query,
            k=3,
            filter={"must": [{"key": "user_id", "match": {"value": user_id}}]}
        )
        if not docs:
            return ""
        facts = "\n".join(f"- {d.page_content}" for d in docs)
        return facts
    except Exception as e:
        logger.error(f"Error recalling preferences: {e}")
        return ""


async def extract_and_save_memories(user_id: str, user_msg: str, assistant_response: str):
    """
    Async LLM extraction: analyze the conversation turn and extract
    any facts/preferences worth remembering long-term.
    Runs as fire-and-forget after the response is sent.
    """
    if not _vector_store or not _llm_client:
        return

    try:
        extraction = await _llm_client.chat.completions.create(
            model=_llm_model,
            messages=[
                {"role": "system", "content": (
                    "You are a memory extraction agent. Analyze the conversation below and extract "
                    "any facts, preferences, or important context worth remembering about this user "
                    "for future conversations. Output ONLY a JSON array of strings, each a concise fact. "
                    "If nothing is worth remembering, output an empty array [].\n\n"
                    "Examples of things to remember:\n"
                    "- User preferences: 'Prefers conservative investment strategy'\n"
                    "- Important context: 'Works at TechCorp as a backend engineer'\n"
                    "- Explicit requests: 'Always send reports in PDF format'\n\n"
                    "DO NOT extract: greetings, tool usage details, temporary instructions, "
                    "or anything already captured by tool outputs (like portfolio paths)."
                )},
                {"role": "user", "content": (
                    f"USER MESSAGE: {user_msg}\n\n"
                    f"ASSISTANT RESPONSE: {assistant_response[:1000]}"
                )},
            ],
            temperature=0,
            max_tokens=300,
        )

        raw = extraction.choices[0].message.content.strip()
        # Parse JSON array
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        memories = json.loads(raw)
        if not memories or not isinstance(memories, list):
            return

        from langchain_core.documents import Document
        docs = [
            Document(
                page_content=mem,
                metadata={
                    "user_id": user_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "llm_extraction",
                }
            )
            for mem in memories
            if isinstance(mem, str) and len(mem.strip()) > 5
        ]
        if docs:
            _vector_store.add_documents(docs)
            logger.info(f"Extracted {len(docs)} memories for user {user_id}: {[d.page_content[:50] for d in docs]}")

    except json.JSONDecodeError:
        logger.debug(f"Memory extraction returned non-JSON, skipping")
    except Exception as e:
        logger.error(f"Memory extraction failed (non-critical): {e}")


# ── Combined Context Builder ─────────────────────────────────────────────────


async def build_memory_context(user_id: str, query: str) -> str:
    """
    Build the full memory context string for injection into the system prompt.
    Combines Layer 1 (facts) + Layer 2 (preferences).
    """
    parts = []

    # Layer 1: Structured facts
    facts = await get_user_facts(user_id)
    if facts:
        fact_lines = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
        parts.append(f"Known facts about this user:\n{fact_lines}")

    # Layer 2: Semantic preferences
    prefs = recall_user_preferences(user_id, query)
    if prefs:
        parts.append(f"Relevant past preferences/context:\n{prefs}")

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts)


# ── Tool Fact Hooks ──────────────────────────────────────────────────────────


async def _hook_portfolio_created(user_id: str, args: Dict, result: Dict):
    """Save portfolio path when rothchild_create_portfolio succeeds."""
    csv_path = result.get("csv_path", "")
    if csv_path:
        await save_user_fact(user_id, "portfolio_path", csv_path)
        await save_user_fact(user_id, "portfolio_date",
                           datetime.datetime.now().strftime("%Y-%m-%d"))
        name = args.get("portfolio_name", "custom")
        await save_user_fact(user_id, "portfolio_name", name)
        count = result.get("holdings_count", 0)
        await save_user_fact(user_id, "portfolio_holdings_count", str(count))


async def _hook_analysis_ran(user_id: str, args: Dict, result: Dict):
    """Save last analysis date when rothchild tools run."""
    status = result.get("status", "")
    if status in ("completed", "enqueued", "created"):
        await save_user_fact(user_id, "last_analysis_date",
                           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


async def _hook_packet_generated(user_id: str, args: Dict, result: Dict):
    """Save that a packet was generated."""
    status = result.get("status", "")
    if status in ("completed", "enqueued", "created"):
        await save_user_fact(user_id, "last_packet_date",
                           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


# Tool name → fact extraction hook
TOOL_FACT_HOOKS: Dict[str, Callable] = {
    "rothchild_create_portfolio": _hook_portfolio_created,
    "rothchild_run": _hook_analysis_ran,
    "rothchild_run_portfolio": _hook_analysis_ran,
    "rothchild_generate_packet": _hook_packet_generated,
}


async def process_tool_facts(user_id: str, tool_name: str, args: Dict[str, Any], result_str: str):
    """
    After a tool executes, check if there's a fact hook and run it.
    Called from tools_node in gateway.py.
    """
    hook = TOOL_FACT_HOOKS.get(tool_name)
    if not hook:
        return

    try:
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
        if isinstance(result, dict) and "error" not in result:
            await hook(user_id, args, result)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        logger.error(f"Fact hook for {tool_name} failed: {e}")
