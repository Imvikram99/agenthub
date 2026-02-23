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
import datetime
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

import instructor
from openai import AsyncOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.hub.registry import ProjectRegistry

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hub_gateway")

# ── Config ─────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HUB_MODEL = os.getenv("HUB_MODEL", "anthropic/claude-sonnet-4")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOOL_ROUNDS = 8

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

def get_user_memory(session_id: str, query: str) -> str:
    """Retrieve relevant memory facts for this session."""
    if not vector_store:
        return ""
        
    try:
        # We filter explicitly by session_id in the payload metadata
        docs = vector_store.similarity_search(
            query=query,
            k=3,
            filter={"must": [{"key": "session_id", "match": {"value": session_id}}]}
        )
        if not docs:
            return ""
        
        facts = "\n".join(f"- {d.page_content}" for d in docs)
        return f"\n\nRelevant past instructions from this user:\n{facts}"
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        return ""

def save_user_memory(session_id: str, fact: str):
    """Save a new explicit preference to Qdrant."""
    if not vector_store:
        return
    try:
        from langchain_core.documents import Document
        doc = Document(
            page_content=fact,
            metadata={"session_id": session_id, "timestamp": datetime.datetime.now().isoformat()}
        )
        vector_store.add_documents([doc])
        logger.info(f"Saved memory for session {session_id}: {fact}")
    except Exception as e:
        logger.error(f"Error saving memory: {e}")


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

class SafeToolCallParameter(BaseModel):
    """
    Pydantic model to enforce safe parameters for CLI/System tools.
    Blocks absolute file paths and shell injections.
    """
    @classmethod
    def validate_safe_arg(cls, v: Any) -> Any:
        if isinstance(v, str):
            # Block obvious absolute path attempts that might traverse to system dirs
            if v.startswith("/") and len(v) > 1 and not v.startswith("/Users/apple/Documents/vikram_workspace"):
                raise ValueError(f"Forbidden absolute path outside workspace: {v}")
            # Block basic subshell injections
            if "$(" in v or "`" in v or "&" in v or "|" in v:
                raise ValueError(f"Potentially unsafe shell characters in argument: {v}")
        return v

def validate_tool_arguments(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Validates arbitrary dictionary arguments against our safety rules."""
    for k, v in args.items():
        SafeToolCallParameter.validate_safe_arg(v)
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
    messages: List[Dict[str, Any]]
    session_id: str


async def agent_node(state: AgentState) -> Dict[str, Any]:
    """Call the LLM with current state."""
    messages = state["messages"]
    session_id = state.get("session_id", "default")
    
    system_prompt = get_system_prompt()
    
    # Retrieve relevant memories and inject them into the system prompt
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if last_user_msg:
        memory_context = get_user_memory(session_id, last_user_msg)
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
        result = await execute_tool(tool_name, tool_args)
        
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


async def agent_chat(session_id: str, user_message: str, history: List[Dict[str, Any]]) -> str:
    """
    Run the LangGraph workflow.
    """
    initial_messages = history + [{"role": "user", "content": user_message}]
    
    # Auto-extract explicit "remember this" or "always do X" instructions 
    # and save them to the vector store
    lower_msg = user_message.lower()
    if "always " in lower_msg or "remember" in lower_msg or "prefer" in lower_msg:
        # A simple heuristic; can be made smarter with LLM memory extraction
        save_user_memory(session_id, user_message)
    
    # We pass recursion_limit equal to MAX_TOOL_ROUNDS + a little buffer
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_TOOL_ROUNDS * 2
    }
    
    try:
        # Run graph
        # Note: we need to pass session_id into the state so the agent_node can use it for memory
        final_state = await app_graph.ainvoke({
            "messages": initial_messages,
            "session_id": session_id
        }, config=config)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        return f"⚠️ Workflow error: {e}"
        
    messages = final_state.get("messages", [])
    if not messages:
        return "No response."
        
    last_message = messages[-1]
    return last_message.get("content", "No text response generated.")


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="Hub Gateway v3", version="3.0.0")

sessions: Dict[str, List[Dict[str, Any]]] = {}


class ChatRequest(BaseModel):
    prompt: str
    context: Dict[str, Any] = {}
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    output: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main entry point. Routes through LLM agent with tool calling."""
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    output = await agent_chat(session_id, req.prompt, history)

    # Save to session history
    history.append({"role": "user", "content": req.prompt})
    history.append({"role": "assistant", "content": output})

    # Trim history if too long
    if len(history) > 40:
        history[:] = history[-30:]

    return ChatResponse(output=output, session_id=session_id)


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


@app.post("/discover")
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


@app.post("/register")
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
