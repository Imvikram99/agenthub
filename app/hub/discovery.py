"""
Discovery Engine — Auto-discovers API endpoints and CLI capabilities
from registered projects and generates OpenAI function-calling tool schemas.

Discovery methods:
  - scan_routes:  Regex parse @app.get/post/patch/delete from Python source
  - manual:       Read explicit endpoint definitions from projects.yaml
  - cli_help:     Parse argparse --help output (future)
  - openapi:      Fetch /openapi.json from running server (future)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hub.discovery")


# ── Route Scanner ──────────────────────────────────────────────────────────────

# Matches: @app.get("/api/v1/leads")  or  @app.post("/api/v1/leads/{lead_id}/stage")
ROUTE_PATTERN = re.compile(
    r"""@app\.(get|post|put|patch|delete)\(\s*["']([^"']+)["']""",
    re.IGNORECASE,
)

# Matches path parameters: {lead_id}, {approval_id}, etc.
PATH_PARAM_PATTERN = re.compile(r"\{(\w+)\}")


def scan_routes(entry_file: str) -> List[Dict[str, Any]]:
    """
    Scan a Python file for FastAPI/Flask route decorators.
    Returns list of discovered endpoints.
    """
    path = Path(entry_file)
    if not path.exists():
        logger.warning(f"Entry file not found: {entry_file}")
        return []

    source = path.read_text()
    endpoints = []

    for match in ROUTE_PATTERN.finditer(source):
        method = match.group(1).upper()
        route_path = match.group(2)

        # Skip non-API routes
        if not route_path.startswith("/api/"):
            continue

        # Extract path parameters
        path_params = PATH_PARAM_PATTERN.findall(route_path)

        # Generate a tool name from the route
        tool_name = _route_to_tool_name(method, route_path)

        # Generate description
        description = _route_to_description(method, route_path)

        endpoints.append({
            "method": method,
            "path": route_path,
            "path_params": path_params,
            "tool_name": tool_name,
            "description": description,
        })

    logger.info(f"Discovered {len(endpoints)} endpoints in {entry_file}")
    return endpoints


def _route_to_tool_name(method: str, path: str) -> str:
    """
    Convert route to tool name.
    POST /api/v1/outreach/drafts:generate → outreach_drafts_generate
    GET /api/v1/leads → leads_list
    PATCH /api/v1/leads/{lead_id}/stage → leads_change_stage
    """
    # Remove /api/v1/ prefix
    clean = re.sub(r"^/api/v\d+/", "", path)
    # Remove path params
    clean = re.sub(r"/\{[^}]+\}", "", clean)
    # Replace colons (action suffixes like :generate)
    clean = clean.replace(":", "_")
    # Replace slashes with underscores
    clean = clean.replace("/", "_").strip("_")

    # Add action verb based on method
    method_prefix = {
        "GET": "list",
        "POST": "create",
        "PUT": "update",
        "PATCH": "update",
        "DELETE": "delete",
    }

    # Don't add prefix if already has action verb in name (like :generate → _generate)
    if "_" in clean and any(clean.endswith(a) for a in [
        "_generate", "_submit", "_apply", "_import", "_ingest",
        "_decision", "_dispatches",
    ]):
        return clean

    # For simple GET endpoints, use "list_" prefix
    if method == "GET":
        return f"list_{clean}"

    return clean


def _route_to_description(method: str, path: str) -> str:
    """Generate a human-readable description from route."""
    return f"{method} {path}"


# ── Tool Schema Generator ─────────────────────────────────────────────────────

def endpoint_to_tool(
    project_name: str,
    endpoint: Dict[str, Any],
    manual_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a discovered endpoint to an OpenAI function-calling tool schema.
    """
    tool_name = f"{project_name}__{endpoint['tool_name']}"
    description = endpoint.get("description", f"{endpoint['method']} {endpoint['path']}")

    # Build parameters
    properties: Dict[str, Any] = {}
    required: List[str] = []

    # Path params are always required
    for p in endpoint.get("path_params", []):
        properties[p] = {"type": "string", "description": f"ID for {p.replace('_', ' ')}"}
        required.append(p)

    # For POST/PATCH, add a generic body parameter unless manual_params given
    if endpoint["method"] in ("POST", "PATCH", "PUT"):
        if manual_params:
            for name, schema in manual_params.items():
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": schema.get("description", name),
                }
                if schema.get("enum"):
                    properties[name]["enum"] = schema["enum"]
                if schema.get("required"):
                    required.append(name)
        else:
            # Generic body for unknown POST endpoints
            properties["body"] = {
                "type": "object",
                "description": "Request body (JSON object)",
            }

    # For GET, add common query params
    if endpoint["method"] == "GET":
        if manual_params:
            for name, schema in manual_params.items():
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": schema.get("description", name),
                }
                if schema.get("enum"):
                    properties[name]["enum"] = schema["enum"]
                if schema.get("required"):
                    required.append(name)

    params_schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        params_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": params_schema,
        },
        "_meta": {
            "project": project_name,
            "method": endpoint["method"],
            "path": endpoint["path"],
            "path_params": endpoint.get("path_params", []),
        },
    }


def cli_command_to_tool(
    project_name: str,
    command: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a CLI command definition to a tool schema."""
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for arg_name, schema in command.get("args", {}).items():
        prop: Dict[str, Any] = {"type": schema.get("type", "string")}
        if schema.get("description"):
            prop["description"] = schema["description"]
        if schema.get("enum"):
            prop["enum"] = schema["enum"]
        if schema.get("required"):
            required.append(arg_name)
        # Boolean flags don't need a value
        if schema.get("flag"):
            prop["type"] = "boolean"
        properties[arg_name.replace("-", "_")] = prop

    params_schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": command["name"],
            "description": command.get("description", command["name"]),
            "parameters": params_schema,
        },
        "_meta": {
            "project": project_name,
            "type": "cli",
            "script": command.get("script", ""),
        },
    }


def custom_tool_to_schema(
    project_name: str,
    tool_def: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a custom tool definition to a tool schema."""
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, schema in tool_def.get("params", {}).items():
        prop: Dict[str, Any] = {"type": schema.get("type", "string")}
        if schema.get("description"):
            prop["description"] = schema["description"]
        if schema.get("enum"):
            prop["enum"] = schema["enum"]
        if schema.get("required"):
            required.append(param_name)
        if schema.get("items"):
            prop["items"] = {"type": "object", "properties": {}}
            for item_name, item_schema in schema["items"].items():
                prop["items"]["properties"][item_name] = {
                    "type": item_schema.get("type", "string"),
                }
                if item_schema.get("enum"):
                    prop["items"]["properties"][item_name]["enum"] = item_schema["enum"]
        properties[param_name] = prop

    params_schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def.get("description", tool_def["name"]),
            "parameters": params_schema,
        },
        "_meta": {
            "project": project_name,
            "type": "custom",
        },
    }
