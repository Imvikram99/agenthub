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

import httpx

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
            "is_hazardous": method in ("POST", "PUT", "PATCH", "DELETE")
        })

    logger.info(f"Discovered {len(endpoints)} endpoints in {entry_file}")
    return endpoints


def fetch_openapi(url: str, headers: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Fetch /openapi.json from a running service and convert it to discovered endpoints.
    Requires httpx to make the network request.
    """
    openapi_url = url.rstrip("/")
    if not openapi_url.endswith("/openapi.json"):
        # Most FastAPI apps have it here by default.
        openapi_url = f"{openapi_url}/openapi.json"

    logger.info(f"Fetching OpenAPI spec from {openapi_url}")
    
    try:
        # We do this synchronously since discovery happens at startup/register time
        # Currently the registry doesn't use async for discovery.
        with httpx.Client(timeout=10.0, headers=headers) as client:
            resp = client.get(openapi_url)
            resp.raise_for_status()
            spec = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch OpenAPI from {openapi_url}: {e}")
        return []

    endpoints = []
    paths = spec.get("paths", {})
    
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            method_upper = method.upper()
            
            # Skip common non-API or internal methods (like OPTIONS/HEAD) unless explicitly wanted.
            if method_upper not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
                
            # Extract basic info
            operation_id = operation.get("operationId", "")
            summary = operation.get("summary", "")
            desc = operation.get("description", result_description_fallback(method_upper, path))
            
            # Prefer operationId for tool_name if available, else generate one
            tool_name = operation_id if operation_id else _route_to_tool_name(method_upper, path)
            
            # Extract parameters
            path_params = []
            query_params = {}
            
            # OpenAPI 3.0 parameters array
            parameters = operation.get("parameters", [])
            for param in parameters:
                if param.get("in") == "path":
                    path_params.append(param.get("name"))
                elif param.get("in") == "query":
                    name = param.get("name")
                    schema = param.get("schema", {})
                    query_params[name] = {
                        "type": schema.get("type", "string"),
                        "description": param.get("description", name),
                        "required": param.get("required", False)
                    }
                    if schema.get("enum"):
                        query_params[name]["enum"] = schema["enum"]

            # Extract Request Body parameters for POST/PUT/PATCH
            body_params = {}
            request_body = operation.get("requestBody", {})
            if request_body:
                content = request_body.get("content", {})
                json_content = content.get("application/json", {})
                body_schema = json_content.get("schema", {})
                
                # Check for $ref in body schema (would need full resolution, 
                # but for simple cases we just log it or pass a generic body)
                if "$ref" in body_schema:
                    ref_path = body_schema["$ref"]
                    # E.g. "#/components/schemas/Item"
                    if ref_path.startswith("#/components/schemas/"):
                        model_name = ref_path.split("/")[-1]
                        components = spec.get("components", {}).get("schemas", {})
                        actual_schema = components.get(model_name, {})
                        
                        properties = actual_schema.get("properties", {})
                        required_fields = actual_schema.get("required", [])
                        
                        for p_name, p_schema in properties.items():
                            body_params[p_name] = {
                                "type": p_schema.get("type", "string"),
                                "description": p_schema.get("description", p_name),
                                "required": p_name in required_fields
                            }
                elif body_schema.get("type") == "object":
                    properties = body_schema.get("properties", {})
                    required_fields = body_schema.get("required", [])
                    
                    for p_name, p_schema in properties.items():
                         body_params[p_name] = {
                            "type": p_schema.get("type", "string"),
                            "description": p_schema.get("description", p_name),
                            "required": p_name in required_fields
                        }

            # Combine query and body params into a single manual_params dict for our generic builder
            combined_params = {**query_params, **body_params}
            
            endpoints.append({
                "method": method_upper,
                "path": path,
                "path_params": path_params,
                "tool_name": tool_name,
                "description": summary or desc,
                "manual_params": combined_params if combined_params else None,
                "is_hazardous": method_upper in ("POST", "PUT", "PATCH", "DELETE")
            })

    logger.info(f"Discovered {len(endpoints)} OpenAPI endpoints from {url}")
    return endpoints

def result_description_fallback(method: str, path: str) -> str:
    return f"{method} {path}"



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

    # If the endpoint came from OpenAPI and provided explicit manual_params, use them.
    if manual_params is None and "manual_params" in endpoint:
        manual_params = endpoint["manual_params"]

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
            "is_hazardous": endpoint.get("is_hazardous", False),
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
