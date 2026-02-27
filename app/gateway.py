#!/usr/bin/env python3
"""
Hub Gateway v3 — Self-evolving LLM-powered tool-calling agent.

Uses OpenRouter (Claude Sonnet / Gemini) with a dynamic project registry.
Projects are defined in projects.yaml and tools are auto-discovered.

Features:
  - Dynamic project registry (projects.yaml)
  - Auto-discovery of API routes and CLI commands
  - LLM-powered intent routing with tool calling
  - Self-evolution via hub_evolve meta-tool
  - Multi-turn conversations with session memory
  - /discover, /projects, /register endpoints

Usage:
    python scripts/hub_gateway.py --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import operator
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

import instructor
from openai import AsyncOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.hub.registry import ProjectRegistry
from app.hub.memory import (
    init_layer2,
    build_memory_context,
    process_tool_facts,
    extract_and_save_memories,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_WORKSPACE = os.getenv("HUB_WORKSPACE_ROOT", "/Users/apple/Documents/vikram_workspace")
HUB_API_KEY = os.getenv("HUB_API_KEY", "")  # Empty = auth disabled (dev mode)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hub_gateway")

# ── Config ─────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HUB_MODEL = os.getenv("HUB_MODEL", "anthropic/claude-sonnet-4")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOOL_ROUNDS = 8
SESSION_TTL_SECONDS = int(os.getenv("HUB_SESSION_TTL", "86400"))  # 24 hours default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Registry Setup ─────────────────────────────────────────────────────────────

registry = ProjectRegistry()
registry.load()


# ── Qdrant Memory Setup ────────────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "hub_user_memory"

vector_store = None
if OPENAI_API_KEY:
    try:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
        if not _qdrant_client.collection_exists(QDRANT_COLLECTION):
            from qdrant_client.http.models import VectorParams, Distance
            _qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        
        # We use OpenAI's standard text-embedding-3-small via LangChain
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = QdrantVectorStore(
            client=_qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding=_embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant Vector Store: {e}")
else:
    logger.warning("OPENAI_API_KEY not set. Long Term Memory (Qdrant) is disabled.")

# Initialize Layer 2 memory with shared Qdrant/LLM instances
# (raw_client is defined later, so we defer init to after LLM setup)


# ── Hub Evolve Meta-Tool ───────────────────────────────────────────────────────

HUB_EVOLVE_TOOL = {
    "type": "function",
    "function": {
        "name": "hub_evolve",
        "description": (
            "Use when the user asks for a feature that NO existing tool supports. "
            "This will create a new branch with the implementation for review. "
            "If you're unsure whether a tool exists, check the available tools first. "
            "Only use this for genuinely missing capabilities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "feature_description": {
                    "type": "string",
                    "description": "What the user wants built (be specific)",
                },
                "target_project": {
                    "type": "string",
                    "description": "Which project should this be added to",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Git branch name for the feature (e.g. hub/slack-alerts)",
                },
            },
            "required": ["feature_description", "target_project"],
        },
    },
}


async def execute_hub_evolve(args: Dict[str, Any]) -> str:
    """
    Self-evolution: use langgraph_run to generate a new feature
    in a target project, on a new git branch.
    """
    feature = args.get("feature_description", "")
    target = args.get("target_project", "")
    branch = args.get("branch_name", f"hub/{target}-evolve-{datetime.datetime.now().strftime('%Y%m%d')}")

    project = registry.projects.get(target)
    if not project:
        available = ", ".join(registry.projects.keys())
        return json.dumps({
            "error": f"Project '{target}' not found. Available: {available}",
        })

    # Find langgraph-poc to dispatch the coding task
    langgraph = registry.projects.get("langgraph-poc")
    if not langgraph:
        return json.dumps({
            "status": "manual_required",
            "message": (
                f"LangGraph coding engine not available. "
                f"Here's what needs to be built:\n\n"
                f"**Feature:** {feature}\n"
                f"**Target:** {target} ({project.root})\n"
                f"**Branch:** {branch}\n\n"
                f"You'll need to implement this manually."
            ),
        })

    # Dispatch to langgraph_run
    prompt = (
        f"In the project at {project.root}, create a new git branch '{branch}' and implement: "
        f"{feature}. "
        f"Make sure the changes are self-contained and well-tested. "
        f"Commit to the branch with a descriptive commit message."
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{langgraph.url}/run",
                json={"message": prompt, "repo": project.root, "branch": branch},
            )
            result = r.json()
    except httpx.ConnectError:
        result = {"error": "LangGraph not running. Start it first."}

    return json.dumps({
        "status": "evolving",
        "feature": feature,
        "target_project": target,
        "branch": branch,
        "langgraph_response": result,
        "message": (
            f"🔧 Creating feature in branch `{branch}`.\n"
            f"Once done, review and merge: `git checkout {branch}`"
        ),
    })


# ── Build Final Tool List ──────────────────────────────────────────────────────

def get_all_tools() -> List[Dict[str, Any]]:
    """Get all tools: registry tools + hub_evolve meta-tool."""
    return registry.tools + [HUB_EVOLVE_TOOL]


def get_system_prompt() -> str:
    """Generate system prompt from registry."""
    return registry.build_system_prompt()


# ── Tool Executor (dispatch to registry or hub_evolve) ─────────────────────────

async def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call. Routes to registry or hub_evolve."""
    if name == "hub_evolve":
        return await execute_hub_evolve(args)
    return await registry.execute_tool(name, args)


# ── Output Structure & Guardrails (Instructor & Pydantic) ────────────────────

BLOCKED_SHELL_PATTERNS = ["$(", "`", "&", "|", ";", "\n", ">>", ">", "<"]


def _validate_safe_arg(v: Any) -> Any:
    """Validate a single string argument for unsafe paths and shell injection."""
    if isinstance(v, str):
        # Block absolute paths outside the workspace (with traversal protection)
        if v.startswith("/") and len(v) > 1:
            resolved = os.path.realpath(v)  # Resolves ../ traversal
            if not resolved.startswith(ALLOWED_WORKSPACE):
                raise ValueError(f"Forbidden path outside workspace: {v} (resolves to {resolved})")
        # Block shell injection patterns
        for pattern in BLOCKED_SHELL_PATTERNS:
            if pattern in v:
                raise ValueError(f"Blocked unsafe pattern '{pattern}' in argument: {v[:100]}")
    return v


def _validate_recursive(value: Any) -> Any:
    """Recursively validate all string values in nested structures."""
    if isinstance(value, str):
        _validate_safe_arg(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _validate_recursive(v)
    elif isinstance(value, list):
        for item in value:
            _validate_recursive(item)
    return value


def validate_tool_arguments(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Validates arbitrary dictionary arguments against safety rules — recursively."""
    _validate_recursive(args)
    return args


# ── OpenRouter LLM Client ─────────────────────────────────────────────────────

raw_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# We use the official OpenAI client patched with Instructor for structured planning
client = instructor.from_openai(
    raw_client,
    mode=instructor.Mode.JSON
)

# Initialize Layer 2 memory now that both vector_store and raw_client exist
init_layer2(vector_store, raw_client, HUB_MODEL)

async def call_llm(
    messages: List[Dict[str, Any]],
    tools: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Call OpenRouter using the native OpenAI SDK (Instructor compatible)."""
    payload: Dict[str, Any] = {
        "model": HUB_MODEL,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    # Use the standard wrapper instead of raw HTTPX to get proper tool calling schema support
    # We use raw_client here because we want raw tool_calls out, not an Instructor Pydantic model
    response = await raw_client.chat.completions.create(**payload)
    
    # Reconstruct the response dict to match the previous httpx format
    # so the rest of the LangGraph node logic doesn't break
    message_dict = {"role": response.choices[0].message.role, "content": response.choices[0].message.content or ""}
    
    if response.choices[0].message.tool_calls:
        message_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in response.choices[0].message.tool_calls
        ]
        
    return {
        "choices": [
            {
                "message": message_dict
            }
        ]
    }


# ── Conversation Engine (LangGraph) ──────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    session_id: str


async def agent_node(state: AgentState) -> Dict[str, Any]:
    """Call the LLM with current state."""
    messages = state["messages"]
    session_id = state.get("session_id", "default")
    
    system_prompt = get_system_prompt()
    
    # Retrieve tiered memory context (Layer 1 facts + Layer 2 preferences)
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if last_user_msg:
        memory_context = await build_memory_context(session_id, last_user_msg)
        if memory_context:
            system_prompt += memory_context

    tools = get_all_tools()

    # Prepend system prompt if not present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = await call_llm(messages, tools)
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"❌ LLM API error: {e}"}]}

    choice = response["choices"][0]
    msg = choice["message"]
    
    return {"messages": [msg]}


async def tools_node(state: AgentState) -> Dict[str, Any]:
    """Execute tools requested by the LLM."""
    messages = state["messages"]
    session_id = state.get("session_id", "default")
    last_message = messages[-1]
    
    tool_calls = last_message.get("tool_calls", [])
    if not tool_calls:
        return {"messages": []}

    results = []
    for tc in tool_calls:
        fn = tc["function"]
        tool_name = fn["name"]
        try:
            tool_args = json.loads(fn.get("arguments", "{}"))
            # Apply strict Pydantic safety validators
            tool_args = validate_tool_arguments(tool_name, tool_args)
        except json.JSONDecodeError:
            tool_args = {}
        except ValueError as ve:
            # Catch Pydantic validation errors (unsafe paths, etc)
            logger.warning(f"Validation blocked tool call to {tool_name}: {ve}")
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tool_name,
                "content": json.dumps({"error": f"Guardrail Validation Error: {ve}"})
            })
            continue

        logger.info(f"Executing tool: {tool_name}({json.dumps(tool_args)[:200]})")
        
        # Human-in-the-loop: Interrupt hazardous tools
        meta = registry.get_tool_meta(tool_name)
        if meta and meta.get("is_hazardous"):
            logger.warning(f"Interrupting for hazardous tool: {tool_name}")
            approval = interrupt({
                "action": "require_approval",
                "tool_name": tool_name,
                "args": tool_args
            })
            if not approval.get("approved"):
                results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": json.dumps({"error": "User denied action execution."})
                })
                continue

        result = await execute_tool(tool_name, tool_args)
        
        # Layer 1: Save structured facts from tool outputs
        await process_tool_facts(session_id, tool_name, tool_args, result)
        
        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tool_name,
            "content": result
        })
        
    return {"messages": results}


def should_continue(state: AgentState) -> str:
    """Determine whether to call tools or end."""
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"
    return END

# Build Graph
memory = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Compile
app_graph = workflow.compile(checkpointer=memory)


async def agent_chat(session_id: str, user_message: str) -> str:
    """
    Run the LangGraph workflow.
    LangGraph's MemorySaver handles message history via thread_id.
    """
    
    # We pass recursion_limit equal to MAX_TOOL_ROUNDS + a little buffer
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_TOOL_ROUNDS * 2
    }
    
    try:
        # Run graph — only pass current user message.
        # LangGraph's MemorySaver persists message history via thread_id.
        final_state = await app_graph.ainvoke({
            "messages": [{"role": "user", "content": user_message}],
            "session_id": session_id
        }, config=config)
        
        # Check if the graph was interrupted
        state_snapshot = await app_graph.aget_state(config)
        if state_snapshot.next:
            # Graph is paused waiting for user input
            interrupt_data = state_snapshot.tasks[0].interrupts[0].value
            action = interrupt_data.get("action")
            tool_name = interrupt_data.get("tool_name")
            
            if action == "require_approval":
                return (
                    f"⚠️ **Approval Required** ⚠️\n\n"
                    f"I need to run a potentially destructive action: `{tool_name}`\n"
                    f"Please approve or deny this action to continue."
                )

    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        return f"⚠️ Workflow error: {e}"
        
    messages = final_state.get("messages", [])
    if not messages:
        return "No response."
    last_message = messages[-1]
    response_text = last_message.get("content", "No text response generated.")
    
    # Layer 2: Async LLM memory extraction (fire-and-forget, post-response)
    asyncio.create_task(extract_and_save_memories(session_id, user_message, response_text))
    
    return response_text


async def stream_chat(session_id: str, user_message: str, context: Optional[Dict[str, Any]] = None):
    """
    Run LangGraph and yield Server-Sent Events (SSE) for real-time frontend streaming.
    LangGraph's MemorySaver handles message history via thread_id.
    """

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_TOOL_ROUNDS * 2
    }

    try:
        # We use astream_events to catch granular internal events (tool starts, LLM tokens)
        async for event in app_graph.astream_events({
            "messages": [{"role": "user", "content": user_message}],
            "session_id": session_id
        }, config=config, version="v1"):
            
            kind = event["event"]
            name = event.get("name", "")
            
            # 1. LLM Token streaming (if the model/OpenAI client supports it)
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    payload = {"event": "model_stream", "content": chunk.content}
                    yield f"data: {json.dumps(payload)}\n\n"
            
            # 2. Tool started executing
            elif kind == "on_tool_start":
                # LangGraph might emit generic tool runnables, filter for our specific tools
                if name != "tools":
                    payload = {"event": "tool_start", "tool_name": name, "args": event["data"].get("input")}
                    yield f"data: {json.dumps(payload)}\n\n"

            # 3. Tool completed
            elif kind == "on_tool_end":
                if name != "tools":
                    # Output might be large, truncate for SSE if necessary
                    output = event["data"].get("output", "")
                    if isinstance(output, str) and len(output) > 500:
                         output = output[:500] + "...[truncated]"
                         
                    payload = {"event": "tool_end", "tool_name": name, "result": output}
                    yield f"data: {json.dumps(payload)}\n\n"
                    
        # Check interrupts at the very end
        state_snapshot = await app_graph.aget_state(config)
        if state_snapshot.next:
            interrupt_data = state_snapshot.tasks[0].interrupts[0].value
            action = interrupt_data.get("action")
            tool_name = interrupt_data.get("tool_name")
            if action == "require_approval":
                payload = {"event": "interrupt", "tool_name": tool_name, "message": "Approval required"}
                yield f"data: {json.dumps(payload)}\n\n"
                return

        # Fetch the very last message for history recording
        final_state = await app_graph.aget_state(config)
        messages = final_state.values.get("messages", [])
        if messages:
             content = messages[-1].get("content")
             log_chat_interaction(session_id, user_message, content, context)
             
             payload = {"event": "done", "session_id": session_id}
             yield f"data: {json.dumps(payload)}\n\n"

    except Exception as e:
        logger.error(f"Stream graph execution failed: {e}")
        payload = {"event": "error", "message": f"Workflow error: {e}"}
        yield f"data: {json.dumps(payload)}\n\n"


# ── FastAPI App ────────────────────────────────────────────────────────────────

def log_chat_interaction(session_id: str, prompt: str, response: str, context: Optional[Dict[str, Any]] = None):
    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        # Store in project_root/logs/YYYY-MM-DD/username.txt
        log_dir = Path(__file__).parent.parent / "logs" / date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = session_id
        if context:
            file_name = context.get("username") or context.get("user_id") or session_id
            
        # Clean up file_name in case of unexpected characters
        file_name = str(file_name).replace("/", "_").replace("\\", "_")
        log_file = log_dir / f"{file_name}.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Session: {session_id}\n")
            if context:
                f.write(f"Context: {context}\n")
            f.write(f"User: {prompt}\n")
            f.write(f"Hub: {response}\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        logger.error(f"Failed to write chat log: {e}")


app = FastAPI(title="Hub Gateway v3", version="3.0.0")

# ── API Key Auth ─────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key if HUB_API_KEY is set. Skip auth in dev mode."""
    if not HUB_API_KEY:
        return  # Auth disabled in dev mode
    if api_key != HUB_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ── Redis Session Store ────────────────────────────────────────────────────────

import redis.asyncio as aioredis

_redis_client: Optional[aioredis.Redis] = None

async def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

async def get_session(session_id: str) -> List[Dict[str, Any]]:
    """Get session history from Redis."""
    try:
        r = await _get_redis()
        data = await r.get(f"hub:session:{session_id}")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Redis session read failed, using empty: {e}")
    return []

async def save_session(session_id: str, history: List[Dict[str, Any]]):
    """Save session history to Redis with TTL."""
    try:
        r = await _get_redis()
        # Keep only last 30 messages to prevent unbounded growth
        trimmed = history[-30:] if len(history) > 30 else history
        await r.set(f"hub:session:{session_id}", json.dumps(trimmed), ex=SESSION_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Redis session write failed: {e}")


class ChatRequest(BaseModel):
    prompt: str
    context: Dict[str, Any] = {}
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    output: str
    session_id: str
    reports: Optional[List[str]] = None
    file_paths: Optional[List[str]] = None  # Absolute paths to files to send as attachments


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """Main entry point. Routes through LLM agent with tool calling."""
    session_id = req.session_id or str(uuid.uuid4())

    output = await agent_chat(session_id, req.prompt)
    
    reports = None
    file_paths = None

    if "[SEND_PORTFOLIO_FILES" in output or "[SEND_PORTFOLIO_REPORTS]" in output:
        # Extract optional job_id from tag: [SEND_PORTFOLIO_FILES:job_id] or [SEND_PORTFOLIO_FILES]
        job_id = None
        import re as _re
        m = _re.search(r"\[SEND_PORTFOLIO_FILES:([a-f0-9]+)\]", output)
        if m:
            job_id = m.group(1)

        output = _re.sub(r"\[SEND_PORTFOLIO_FILES[^\]]*\]", "", output)
        output = output.replace("[SEND_PORTFOLIO_REPORTS]", "").strip()
        output = output or "✅ Portfolio analysis complete! Sending reports..."

        # Look up when this job was queued — only accept files written after that time
        not_before = None
        if job_id:
            try:
                r = await _get_redis()
                ts = await r.get(f"hub:job:{job_id}:queued_at")
                if ts:
                    not_before = float(ts)
            except Exception:
                pass

        # Find result files, filtered by job queue time to prevent cross-user mix-up
        file_paths = _find_latest_result_files(not_before=not_before)
        if file_paths:
            # Archive the result files to a user-scoped permanent directory
            rotchild_project = registry.projects.get("rothchild")
            if rotchild_project:
                archive_dir = await _archive_result_files(
                    file_paths, session_id, rotchild_project.root
                )
                if archive_dir:
                    # Save to Layer 1 memory facts for future resend
                    from app.hub.memory import save_user_fact
                    await save_user_fact(session_id, "last_result_dir", archive_dir)
                    await save_user_fact(session_id, "last_result_date",
                                       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        else:
            output += "\n\n⚠️ Could not find result files. The analysis may still be running — try asking again in a minute."

    elif "[GET_LAST_REPORTS]" in output:
        # User wants to resend the last analysis — no re-run needed
        output = output.replace("[GET_LAST_REPORTS]", "").strip() or "Here are your last portfolio reports:"
        from app.hub.memory import get_user_facts
        facts = await get_user_facts(session_id)
        last_dir = facts.get("last_result_dir", "")
        if last_dir and os.path.isdir(last_dir):
            file_paths = [
                os.path.join(last_dir, f)
                for f in ["result.md", "result2.md"]
                if os.path.exists(os.path.join(last_dir, f))
            ]
            last_date = facts.get("last_result_date", "unknown date")
            output = f"📄 Resending your last analysis ({last_date}):"
        else:
            output = "⚠️ No previous analysis found. Please share your portfolio holdings and I'll run a fresh analysis."

    # Save to session history (for logging/portfolio reports)
    history = await get_session(session_id)
    history.append({"role": "user", "content": req.prompt})
    history.append({"role": "assistant", "content": output})
    await save_session(session_id, history)

    log_chat_interaction(session_id, req.prompt, output, req.context)

    return ChatResponse(output=output, session_id=session_id, reports=reports, file_paths=file_paths)


def _find_latest_result_files(not_before: Optional[float] = None) -> List[str]:
    """Find freshest result.md and result2.md from rothchild.
    
    Args:
        not_before: Unix timestamp. If set, only return files modified at or after this time.
                    This prevents multi-user race conditions where User A picks up User B's results.
    """
    import glob
    rothchild_project = registry.projects.get("rothchild")
    if not rothchild_project:
        return []
    result_base = os.path.join(rothchild_project.root, "data", "result")
    if not os.path.exists(result_base):
        return []
    subdirs = [d for d in glob.glob(os.path.join(result_base, "*")) if os.path.isdir(d)]
    if not subdirs:
        return []
    # Pick the subdir with the most recently modified result file
    latest = max(subdirs, key=os.path.getmtime)
    paths = []
    for fname in ["result.md", "result2.md"]:
        fpath = os.path.join(latest, fname)
        if not os.path.exists(fpath):
            continue
        # Guard: only accept files written after this job was queued
        if not_before is not None and os.path.getmtime(fpath) < not_before:
            logger.warning(f"Skipping stale result file (mtime predates job queue): {fpath}")
            continue
        paths.append(fpath)
    return paths


async def _archive_result_files(file_paths: List[str], user_id: str, rothchild_root: str) -> str:
    """Copy result files to a user-scoped archive directory for permanent storage.

    Returns the archive directory path, or empty string on failure.
    Archive structure: data/archive/{user_id}/{YYYYMMDD_HHMMSS}/
    Each user gets isolated storage — multiple portfolios accumulate over time.
    """
    import shutil
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(rothchild_root, "data", "archive", str(user_id), ts)
        os.makedirs(archive_dir, exist_ok=True)
        for fpath in file_paths:
            dest = os.path.join(archive_dir, os.path.basename(fpath))
            shutil.copy2(fpath, dest)
            logger.info(f"Archived {os.path.basename(fpath)} → {archive_dir}")
        return archive_dir
    except Exception as e:
        logger.error(f"Failed to archive result files: {e}")
        return ""


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest):
    """Event Stream (SSE) entry point for UI clients requiring real-time updates."""
    session_id = req.session_id or str(uuid.uuid4())

    return StreamingResponse(
        stream_chat(session_id, req.prompt, req.context),
        media_type="text/event-stream"
    )

class ResumeRequest(BaseModel):
    approved: bool


@app.post("/chat/resume/{thread_id}", dependencies=[Depends(verify_api_key)])
async def resume_chat(thread_id: str, req: ResumeRequest):
    """Resume a paused graph execution with an approval decision."""
    from langgraph.types import Command
    config = {"configurable": {"thread_id": thread_id}}
    
    # Inject the approval decision back into the interrupt point
    final_state = await app_graph.ainvoke(
        Command(resume={"approved": req.approved}),
        config=config
    )
    
    # If it was interrupted AGAIN (unlikely but possible)
    state_snapshot = await app_graph.aget_state(config)
    if state_snapshot.next:
        interrupt_data = state_snapshot.tasks[0].interrupts[0].value
        return {
            "output": f"⚠️ **Approval Required** for `{interrupt_data.get('tool_name')}`",
            "session_id": thread_id
        }

    messages = final_state.get("messages", [])
    if not messages:
        return {"output": "No response after resume.", "session_id": thread_id}
        
    last_message = messages[-1]
    return {
        "output": last_message.get("content", "No text response generated after resume."),
        "session_id": thread_id
    }


@app.get("/health")
async def health():
    """Check health of hub and all backends."""
    result = {
        "hub": "up",
        "model": HUB_MODEL,
        "api_key_set": bool(OPENROUTER_API_KEY),
        "projects": {},
        "total_tools": len(get_all_tools()),
    }

    for name, project in registry.projects.items():
        if project.type == "api" and project.url:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    # Try common health endpoints
                    for ep in ["/healthz", "/health", "/api/v1/health", "/"]:
                        try:
                            r = await client.get(f"{project.url}{ep}")
                            if r.status_code < 500:
                                result["projects"][name] = {"status": "up", "url": project.url}
                                break
                        except Exception:
                            continue
                    else:
                        result["projects"][name] = {"status": "down", "url": project.url}
            except Exception:
                result["projects"][name] = {"status": "down", "url": project.url}
        elif project.type == "cli":
            py_bin = project.python_bin()
            result["projects"][name] = {
                "status": "ready" if Path(py_bin).exists() else "missing venv",
                "root": project.root,
            }

    return result


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    tools = get_all_tools()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", "")[:100],
            }
            for t in tools
        ],
    }


@app.get("/projects")
async def list_projects():
    """List all registered projects and their tools."""
    return {"projects": registry.project_summary()}


@app.post("/discover", dependencies=[Depends(verify_api_key)])
async def discover(project_name: Optional[str] = None):
    """Re-run discovery for one or all projects."""
    results = registry.rediscover(project_name)
    return {
        "discovered": results,
        "total_tools": len(get_all_tools()),
    }


class RegisterRequest(BaseModel):
    name: str
    type: str = "api"
    root: str = ""
    url: str = ""
    port: Optional[int] = None
    description: str = ""
    discovery: Dict[str, Any] = {}


@app.post("/register", dependencies=[Depends(verify_api_key)])
async def register_project(req: RegisterRequest):
    """Register a new project at runtime."""
    config = {
        "type": req.type,
        "root": req.root,
        "url": req.url,
        "port": req.port,
        "description": req.description,
        "discovery": req.discovery,
    }

    tools_count = registry.register_project(req.name, config)
    return {
        "status": "registered",
        "project": req.name,
        "tools_discovered": tools_count,
        "total_tools": len(get_all_tools()),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    all_tools = get_all_tools()

    print(f"\n🌐 Hub Gateway v3 starting on http://{args.host}:{args.port}")
    print(f"   Model: {HUB_MODEL}")
    print(f"   API Key: {'✅ set' if OPENROUTER_API_KEY else '❌ missing'}")
    print(f"   Projects: {len(registry.projects)}")
    for name, p in registry.projects.items():
        loc = p.url or p.root
        print(f"     {name:18s} → {loc} ({len(p.tools)} tools)")
    print(f"   Total Tools: {len(all_tools)} ({len(all_tools) - 1} + hub_evolve)")

    uvicorn.run(app, host=args.host, port=args.port)
