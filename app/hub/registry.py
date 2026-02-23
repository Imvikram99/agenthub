"""
Project Registry — Loads projects.yaml, runs discovery, manages dynamic tools.

Responsibilities:
  - Load project definitions from YAML
  - Run discovery for each project (scan_routes, manual, cli)
  - Build unified tool list for the LLM
  - Generate system prompt dynamically from registered projects
  - Provide generic HTTP/CLI tool executors
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from arq import create_pool
from arq.connections import RedisSettings

from .discovery import (
    scan_routes,
    fetch_openapi,
    endpoint_to_tool,
    cli_command_to_tool,
    custom_tool_to_schema,
)

logger = logging.getLogger("hub.registry")

PROJECTS_YAML = Path(__file__).parent.parent.parent / "projects.yaml"


class Project:
    """Represents a registered project."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.type = config.get("type", "api")
        self.root = config.get("root", "")
        self.url = config.get("url", "")
        self.port = config.get("port")
        self.venv = config.get("venv", ".venv")
        self.description = config.get("description", "")
        self.discovery = config.get("discovery", {})
        self.default_headers = self.discovery.get("default_headers", {})
        self.custom_tools_config = config.get("custom_tools", [])
        self.manual_tools_config = self.discovery.get("manual_tools", [])
        self.tools: List[Dict[str, Any]] = []

    def discover(self) -> List[Dict[str, Any]]:
        """Run discovery and return tool schemas."""
        method = self.discovery.get("method", "manual")
        tools = []

        if method == "scan_routes":
            entry = os.path.join(self.root, self.discovery.get("entry_file", ""))
            endpoints = scan_routes(entry)
            for ep in endpoints:
                tool = endpoint_to_tool(self.name, ep)
                tools.append(tool)
            logger.info(f"[{self.name}] scan_routes discovered {len(tools)} tools")

        elif method == "openapi":
            endpoints = fetch_openapi(self.url, headers=self.default_headers)
            for ep in endpoints:
                tool = endpoint_to_tool(self.name, ep)
                tools.append(tool)
            logger.info(f"[{self.name}] openapi discovered {len(endpoints)} tools")

        elif method == "manual":
            # API endpoints
            for ep_def in self.discovery.get("endpoints", []):
                ep = {
                    "method": ep_def.get("method", "POST"),
                    "path": ep_def["path"],
                    "path_params": [],
                    "tool_name": ep_def.get("tool_name", ep_def["path"]),
                    "description": ep_def.get("description", ""),
                }
                params = ep_def.get("params", {})
                tool = endpoint_to_tool(self.name, ep, manual_params=params)
                # Use the explicit tool_name if provided
                if ep_def.get("tool_name"):
                    tool["function"]["name"] = ep_def["tool_name"]
                tools.append(tool)

            # CLI commands
            for cmd in self.discovery.get("commands", []):
                tool = cli_command_to_tool(self.name, cmd)
                tools.append(tool)

            logger.info(f"[{self.name}] manual config: {len(tools)} tools")

        # Add manual tool overrides (always added, even with scan_routes)
        for mt in self.manual_tools_config:
            tool = endpoint_to_tool(self.name, {
                "method": mt["endpoint"].split()[0],
                "path": mt["endpoint"].split()[1],
                "path_params": [],
                "tool_name": mt["name"],
                "description": mt.get("description", ""),
            }, manual_params=mt.get("params", {}))
            tool["function"]["name"] = mt["name"]
            tools.append(tool)

        # Add custom tools
        for ct in self.custom_tools_config:
            tool = custom_tool_to_schema(self.name, ct)
            tools.append(tool)

        self.tools = tools
        return tools

    def python_bin(self) -> str:
        """Get path to project's Python binary."""
        return os.path.join(self.root, self.venv, "bin/python")

    def build_env(self) -> Dict[str, str]:
        """Build environment dict for subprocess."""
        env = os.environ.copy()
        venv_path = os.path.join(self.root, self.venv)
        venv_bin = os.path.join(venv_path, "bin")
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
        env["VIRTUAL_ENV"] = venv_path
        env_file = os.path.join(self.root, ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return env


class ProjectRegistry:
    """Manages all registered projects and their tools."""

    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or PROJECTS_YAML
        self.projects: Dict[str, Project] = {}
        self.tools: List[Dict[str, Any]] = []
        self._tool_meta: Dict[str, Dict[str, Any]] = {}  # tool_name → metadata
        self._redis_pool = None

    def load(self) -> None:
        """Load projects from YAML and run discovery."""
        if not self.yaml_path.exists():
            logger.error(f"Projects file not found: {self.yaml_path}")
            return

        with open(self.yaml_path) as f:
            config = yaml.safe_load(f)

        for name, proj_config in config.get("projects", {}).items():
            project = Project(name, proj_config)
            project.discover()
            self.projects[name] = project

        self._rebuild_tools()

        # Add the generic global tool for checking ARQ task status
        self.tools.append({
            "type": "function",
            "function": {
                "name": "check_task_status",
                "description": "Check the status of a background CLI task using its queued job ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The job ID returned when the CLI tool was launched."
                        }
                    },
                    "required": ["task_id"]
                }
            }
        })
        # Note: we handle it directly in execute_tool, so no _tool_meta entry is strictly required
        # but we'll add one so execute_tool knows how to route it.
        self._tool_meta["check_task_status"] = {"type": "system_global"}

        logger.info(f"Registry loaded: {len(self.projects)} projects, {len(self.tools)} tools")

    def _rebuild_tools(self) -> None:
        """Rebuild unified tool list from all projects, deduplicating by name."""
        seen: Dict[str, int] = {}  # tool_name → index in self.tools
        self.tools = []
        self._tool_meta = {}
        for project in self.projects.values():
            for tool in project.tools:
                meta = tool.get("_meta", {}).copy()
                meta["project_name"] = project.name
                meta["project_root"] = project.root
                meta["project_url"] = project.url
                meta["project_type"] = project.type
                meta["default_headers"] = project.default_headers

                # Build clean tool dict (without _meta) for LLM
                clean_tool = {
                    "type": tool["type"],
                    "function": tool["function"],
                }

                name = tool["function"]["name"]

                if name in seen:
                    # Override duplicate with later definition
                    idx = seen[name]
                    self.tools[idx] = clean_tool
                else:
                    seen[name] = len(self.tools)
                    self.tools.append(clean_tool)

                self._tool_meta[name] = meta

    def get_tool_meta(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a tool (project, method, path, etc.)."""
        return self._tool_meta.get(tool_name)

    def register_project(self, name: str, config: Dict[str, Any]) -> int:
        """Register a new project at runtime. Returns number of tools discovered."""
        project = Project(name, config)
        discovered = project.discover()
        self.projects[name] = project
        self._rebuild_tools()

        # Persist to YAML
        self._save_yaml()
        return len(discovered)

    def _save_yaml(self) -> None:
        """Persist current registry to YAML."""
        config = {"projects": {}}
        # Re-read existing to preserve comments/formatting
        if self.yaml_path.exists():
            with open(self.yaml_path) as f:
                config = yaml.safe_load(f) or {"projects": {}}

        # We only add new projects, don't overwrite existing
        for name, project in self.projects.items():
            if name not in config.get("projects", {}):
                config.setdefault("projects", {})[name] = {
                    "type": project.type,
                    "root": project.root,
                    "url": project.url,
                    "port": project.port,
                    "description": project.description,
                    "discovery": project.discovery,
                }

        with open(self.yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def rediscover(self, project_name: Optional[str] = None) -> Dict[str, int]:
        """Re-run discovery for one or all projects."""
        results = {}
        targets = [project_name] if project_name else list(self.projects.keys())
        for name in targets:
            if name in self.projects:
                tools = self.projects[name].discover()
                results[name] = len(tools)
        self._rebuild_tools()
        return results

    def build_system_prompt(self) -> str:
        """Generate system prompt dynamically from registered projects."""
        sections = [
            "You are the Hub — a unified command center that manages projects via Telegram.\n",
            "## Your Projects\n",
        ]

        for i, (name, project) in enumerate(self.projects.items(), 1):
            tool_names = [t["function"]["name"] for t in project.tools]
            sections.append(f"### {i}. {name}")
            sections.append(f"- Description: {project.description.strip()}")
            sections.append(f"- Type: {project.type}")
            if tool_names:
                sections.append(f"- Tools: {', '.join(f'`{t}`' for t in tool_names)}")
            sections.append("")

        sections.append("""## Behavioral Rules

1. **Ask before acting** — If the user's request is ambiguous, ask clarifying questions. Don't guess.
2. **Chain when needed** — Some tasks require multiple tools in sequence.
3. **Explain what you did** — After tool calls, summarize results in plain language.
4. **Handle errors gracefully** — If a backend is down, tell the user and suggest how to fix it.
5. **Be concise** — This is Telegram. Short messages. Use emoji for status indicators.
6. **Don't fabricate data** — Only report what the tools actually return.
7. **When no tool exists** — If the user asks for something no tool supports, use `hub_evolve` to suggest building it.

## Rothchild Portfolio Workflow
When a user mentions stocks or holdings (e.g. "I have 10 shares of HDFC"), follow this flow:
1. Gather holdings: Ask for symbol, quantity, avg cost, and exchange if not provided.
2. Create portfolio: Use `rothchild_create_portfolio` to generate a CSV from the holdings
3. Run analysis: Use `rothchild_run_portfolio` with the generated CSV path
4. Read results: Use `rothchild_read_log` to get the analysis output

If user provides a CSV file path directly, skip to step 3.
If user just says "run analysis" without specifying holdings, use `rothchild_run` (uses default sample portfolio).
""")

        return "\n".join(sections)

    def project_summary(self) -> List[Dict[str, Any]]:
        """Return project info for /projects endpoint."""
        return [
            {
                "name": name,
                "type": p.type,
                "root": p.root,
                "url": p.url,
                "description": p.description.strip(),
                "tools_count": len(p.tools),
                "tool_names": [t["function"]["name"] for t in p.tools],
            }
            for name, p in self.projects.items()
        ]

    # ── Generic Tool Executor ──────────────────────────────────────────────────

    async def get_redis(self):
        """Lazy load Redis ARQ connection pool."""
        if self._redis_pool is None:
            self._redis_pool = await create_pool(RedisSettings())
        return self._redis_pool

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Generic tool executor. Uses tool metadata to decide how to dispatch.
        Falls back to specific handlers for custom tools.
        """
        meta = self.get_tool_meta(tool_name)
        if not meta:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        project_name = meta["project_name"]
        project = self.projects.get(project_name)
        if not project:
            return json.dumps({"error": f"Project not found: {project_name}"})

        try:
            # Handle strictly global hub tools
            if meta.get("type") == "system_global":
                if tool_name == "check_task_status":
                    return await self._execute_check_task_status(args)
                return json.dumps({"error": f"Unknown global tool: {tool_name}"})

            # Custom tools need specific handlers
            if meta.get("type") == "custom":
                return await self._execute_custom(tool_name, args, project, meta)

            # CLI tools
            if meta.get("type") == "cli":
                return await self._execute_cli(tool_name, args, project, meta)

            # API tools (default)
            return await self._execute_api(tool_name, args, project, meta)

        except httpx.ConnectError:
            return json.dumps({
                "error": f"{project_name} backend not running.",
                "suggestion": f"Start it: cd {project.root} && {self._start_command(project)}",
            })
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed")
            return json.dumps({"error": f"Tool {tool_name} failed: {str(e)}"})

    async def _execute_api(
        self, tool_name: str, args: Dict[str, Any],
        project: Project, meta: Dict[str, Any],
    ) -> str:
        """Execute an API tool via HTTP."""
        method = meta.get("method", "GET")
        path_template = meta.get("path", "/")

        # Substitute path parameters
        path = path_template
        for param in meta.get("path_params", []):
            if param in args:
                path = path.replace(f"{{{param}}}", str(args.pop(param)))

        headers = {**project.default_headers}

        async with httpx.AsyncClient(
            base_url=project.url, timeout=120, headers=headers,
        ) as client:
            if method == "GET":
                r = await client.get(path, params=args or None)
            elif method in ("POST", "PUT"):
                # Add idempotency key for POST
                if method == "POST":
                    headers["Idempotency-Key"] = str(uuid.uuid4())
                body = args.get("body", args)
                r = await client.post(path, json=body, headers=headers)
            elif method == "PATCH":
                body = args.get("body", args)
                r = await client.patch(path, json=body)
            elif method == "DELETE":
                r = await client.delete(path)
            else:
                return json.dumps({"error": f"Unsupported method: {method}"})

            return json.dumps(r.json(), indent=2, default=str)

    async def _execute_cli(
        self, tool_name: str, args: Dict[str, Any],
        project: Project, meta: Dict[str, Any],
    ) -> str:
        """Execute a CLI tool as an ARQ asynchronous background task."""
        script = meta.get("script", "")
        python_bin = project.python_bin()

        log_dir = os.path.join(project.root, "data/openclaw_runs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"run_{ts}.log")

        env = project.build_env()

        # Build command
        if script.startswith("-m"):
            cmd = [python_bin] + script.split()
        else:
            cmd = [python_bin, script]

        # Add args as CLI flags
        for key, value in args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        # Dispatch background job to ARQ instead of spawning inline
        try:
            redis = await self.get_redis()
            # Enqueue the job for our worker
            job = await redis.enqueue_job(
                "run_cli_command",
                cmd=cmd,
                env=env,
                cwd=project.root,
                log_path=log_path
            )
            
            return json.dumps({
                "status": "queued",
                "task_id": job.job_id,
                "log_path": log_path,
                "command": " ".join(cmd),
                "message": f"Job queued to worker. ID: {job.job_id}",
            })
        except Exception as e:
            logger.error(f"Failed to queue task to ARQ: {e}")
            return json.dumps({
                "error": f"Task Queue error: {e}. Is Redis and the worker running?"
            })

    async def _execute_check_task_status(self, args: Dict[str, Any]) -> str:
        """Check status of an ARQ job."""
        task_id = args.get("task_id")
        if not task_id:
            return json.dumps({"error": "task_id is required"})

        try:
            redis = await self.get_redis()
            from arq.jobs import Job
            job = Job(task_id, redis)
            
            status = await job.status()
            info = await job.info()
            
            result_payload = {
                "task_id": task_id,
                "status": status.value,
            }
            
            if status.value == "complete":
                result = await job.result()
                result_payload.update(result)
            
            return json.dumps(result_payload, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to check task status: {e}")
            return json.dumps({"error": str(e)})

    async def _execute_custom(
        self, tool_name: str, args: Dict[str, Any],
        project: Project, meta: Dict[str, Any],
    ) -> str:
        """Execute custom tools that need special logic."""
        import csv as csv_mod

        if tool_name == "rothchild_create_portfolio":
            holdings = args.get("holdings", [])
            if not holdings:
                return json.dumps({"error": "No holdings provided"})

            portfolio_name = args.get("portfolio_name", "custom")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            portfolio_dir = os.path.join(project.root, "data/portfolios")
            os.makedirs(portfolio_dir, exist_ok=True)
            csv_path = os.path.join(portfolio_dir, f"{portfolio_name}_{ts}.csv")

            with open(csv_path, "w", newline="") as f:
                writer = csv_mod.DictWriter(
                    f,
                    fieldnames=["symbol", "exchange", "quantity", "avg_cost", "current_price", "buy_date"],
                )
                writer.writeheader()
                for h in holdings:
                    sym = h.get("symbol", "")
                    writer.writerow({
                        "symbol": sym.upper() if isinstance(sym, str) else sym,
                        "exchange": h.get("exchange", "NSE"),
                        "quantity": h.get("quantity", 0),
                        "avg_cost": h.get("avg_cost", 0),
                        "current_price": h.get("current_price", 0),
                        "buy_date": h.get("buy_date", ""),
                    })

            return json.dumps({
                "status": "created",
                "csv_path": csv_path,
                "holdings_count": len(holdings),
                "message": f"Portfolio CSV created. Use rothchild_run_portfolio to analyze it.",
            })

        elif tool_name == "rothchild_read_log":
            log_dir = os.path.join(project.root, "data/openclaw_runs")
            if not os.path.isdir(log_dir):
                return json.dumps({"error": "No run logs found yet"})
            logs = sorted(Path(log_dir).glob("run_*.log"), reverse=True)
            if not logs:
                return json.dumps({"error": "No run logs found"})
            content = logs[0].read_text()
            if len(content) > 3000:
                content = content[:1500] + "\n...[truncated]...\n" + content[-1500:]
            return json.dumps({"log_file": str(logs[0]), "content": content})

        return json.dumps({"error": f"No custom handler for: {tool_name}"})

    def _start_command(self, project: Project) -> str:
        """Generate start command for a project."""
        if project.type == "api":
            return f"uvicorn app.main:app --port {project.port or 8000}"
        return "python -m src.runner"
