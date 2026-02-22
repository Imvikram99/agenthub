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
from pydantic import BaseModel

from app.hub.registry import ProjectRegistry

load_dotenv()

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


# ── OpenRouter LLM Client ─────────────────────────────────────────────────────

async def call_llm(
    messages: List[Dict[str, Any]],
    tools: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Call OpenRouter chat completions with function calling support."""
    payload: Dict[str, Any] = {
        "model": HUB_MODEL,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if r.status_code != 200:
            raise Exception(f"OpenRouter API error ({r.status_code}): {r.text}")
        return r.json()


# ── Conversation Engine ───────────────────────────────────────────────────────

async def agent_chat(user_message: str, history: List[Dict[str, Any]]) -> str:
    """
    Run the agent loop:
    1. Send user message + tools to LLM
    2. If LLM wants to call tools → execute them, feed results back
    3. Repeat until LLM returns a text response (or max rounds)
    """
    system_prompt = get_system_prompt()
    tools = get_all_tools()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        logger.info(f"LLM round {round_num}")

        try:
            response = await call_llm(messages, tools)
        except Exception as e:
            return f"❌ LLM API error: {e}"

        choice = response["choices"][0]
        msg = choice["message"]

        # If LLM returns text (no tool calls), we're done
        if msg.get("content") and not msg.get("tool_calls"):
            return msg["content"]

        # If LLM wants to call tools
        if msg.get("tool_calls"):
            messages.append(msg)  # assistant message with tool_calls

            for tc in msg["tool_calls"]:
                fn = tc["function"]
                tool_name = fn["name"]
                try:
                    tool_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                logger.info(f"Executing tool: {tool_name}({json.dumps(tool_args)[:200]})")
                result = await execute_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            continue

        # Fallback: return whatever content we got
        return msg.get("content", "No response from LLM.")

    return "⚠️ Reached maximum tool rounds. Please try a simpler request."


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

    output = await agent_chat(req.prompt, history)

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
