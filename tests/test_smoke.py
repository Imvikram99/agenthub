"""
Smoke test for Agent Hub — validates imports, registry loading, 
tool discovery, guardrails, and session helpers.
"""

import json
import os
import sys
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    """All core modules should import without error."""
    from app.hub.discovery import scan_routes, endpoint_to_tool, cli_command_to_tool
    from app.hub.registry import ProjectRegistry, Project


def test_registry_loads():
    """Registry should load projects.yaml and discover tools."""
    from app.hub.registry import ProjectRegistry
    registry = ProjectRegistry()
    registry.load()
    assert len(registry.projects) > 0, "No projects loaded"
    assert len(registry.tools) > 0, "No tools discovered"


def test_tool_names_unique():
    """All tool names should be unique after deduplication."""
    from app.hub.registry import ProjectRegistry
    registry = ProjectRegistry()
    registry.load()
    names = [t["function"]["name"] for t in registry.tools]
    assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"


def test_guardrails_block_unsafe_path():
    """Guardrails should block paths outside workspace."""
    from app.gateway import validate_tool_arguments
    with pytest.raises(ValueError, match="Forbidden path"):
        validate_tool_arguments("test_tool", {"path": "/etc/passwd"})


def test_guardrails_block_traversal():
    """Guardrails should block path traversal attacks."""
    from app.gateway import validate_tool_arguments, ALLOWED_WORKSPACE
    # Construct a traversal path that starts inside workspace but escapes
    traversal = os.path.join(ALLOWED_WORKSPACE, "..", "..", "etc", "passwd")
    with pytest.raises(ValueError, match="Forbidden path"):
        validate_tool_arguments("test_tool", {"path": traversal})


def test_guardrails_block_shell_injection():
    """Guardrails should block shell injection patterns."""
    from app.gateway import validate_tool_arguments
    dangerous_inputs = [
        {"cmd": "ls; rm -rf /"},
        {"cmd": "cat $(whoami)"},
        {"cmd": "echo `id`"},
        {"cmd": "ls | grep foo"},
    ]
    for args in dangerous_inputs:
        with pytest.raises(ValueError, match="Blocked unsafe pattern"):
            validate_tool_arguments("test_tool", args)


def test_guardrails_recursive():
    """Guardrails should validate nested structures recursively."""
    from app.gateway import validate_tool_arguments
    with pytest.raises(ValueError):
        validate_tool_arguments("test_tool", {
            "holdings": [{"symbol": "AAPL", "path": "/etc/shadow"}]
        })


def test_guardrails_allow_safe_input():
    """Guardrails should pass safe inputs through."""
    from app.gateway import validate_tool_arguments, ALLOWED_WORKSPACE
    safe_path = os.path.join(ALLOWED_WORKSPACE, "mylanggraph", "agenthub")
    result = validate_tool_arguments("test_tool", {
        "prompt": "Create a hello world file",
        "repo_path": safe_path
    })
    assert result["prompt"] == "Create a hello world file"


def test_discovery_scan_routes():
    """scan_routes should handle missing files gracefully."""
    from app.hub.discovery import scan_routes
    result = scan_routes("/nonexistent/file.py")
    assert result == []


def test_discovery_endpoint_to_tool():
    """endpoint_to_tool should produce valid OpenAI function schema."""
    from app.hub.discovery import endpoint_to_tool
    tool = endpoint_to_tool("test_project", {
        "method": "GET",
        "path": "/api/v1/items",
        "path_params": [],
        "tool_name": "list_items",
        "description": "List all items",
    })
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "test_project__list_items"
    assert "parameters" in tool["function"]


def test_memory_imports():
    """Memory module should import without error."""
    from app.hub.memory import (
        save_user_fact, get_user_facts, delete_user_fact,
        recall_user_preferences, extract_and_save_memories,
        build_memory_context, process_tool_facts, TOOL_FACT_HOOKS,
    )


def test_tool_fact_hooks_exist():
    """Key tools should have fact extraction hooks."""
    from app.hub.memory import TOOL_FACT_HOOKS
    expected = ["rothchild_create_portfolio", "rothchild_run", "rothchild_generate_packet"]
    for tool in expected:
        assert tool in TOOL_FACT_HOOKS, f"Missing fact hook for {tool}"


@pytest.mark.asyncio
async def test_save_and_get_user_facts():
    """Layer 1: Save and retrieve structured user facts via Redis."""
    from app.hub.memory import save_user_fact, get_user_facts, delete_user_fact
    test_user = "__test_user_smoke__"
    try:
        await save_user_fact(test_user, "test_key", "test_value")
        facts = await get_user_facts(test_user)
        assert facts.get("test_key") == "test_value"
    finally:
        await delete_user_fact(test_user, "test_key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
