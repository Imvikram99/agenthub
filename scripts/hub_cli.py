#!/usr/bin/env python3
import sys
import os
import asyncio
import logging
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Ensure relative imports work when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.cli_agent import cli_graph, CLIState

console = Console()

async def run_cli_agent(prompt: str):
    console.print(Panel(f"🧠 **Agent Hub Multi-Agent Planner**\n\nGoal: {prompt}", style="bold cyan"))
    
    initial_state = {
        "original_prompt": prompt,
        "upgraded_prompt": "",
        "plan": [],
        "current_task_index": 0,
        "worker_messages": [],
        "completed_steps": [],
        "final_report": "",
        "retry_count": 0,
        "llm_call_count": 0
    }

    try:
        # Stream the graph execution to print live updates
        config = {"recursion_limit": 100}
        
        async for output in cli_graph.astream(initial_state, config=config):
            for node_name, state_update in output.items():
                if node_name == "upgrade_prompt":
                    console.print("\n✨ **Prompt Upgrader:**", style="bold magenta")
                    console.print(state_update.get("upgraded_prompt", "")[:500] + "...\n")
                
                elif node_name == "create_plan":
                    plan = state_update.get("plan", [])
                    console.print(f"\n📋 **Planner Agent created {len(plan)} subtasks:**", style="bold yellow")
                    for t in plan:
                        console.print(f"  {t.id}. {t.description}")
                
                elif node_name == "run_subtask":
                    # Only print if we actually attempted executing something this loop
                    if "worker_messages" in state_update and state_update["worker_messages"]:
                        msg = state_update["worker_messages"][0]
                        if "tool_calls" in msg:
                            for tc in msg["tool_calls"]:
                                fn = tc["function"]["name"]
                                args = tc["function"].get("arguments", "{}")
                                console.print(f"  🛠️ **Worker Executing Tool:** {fn}", style="bold green")
                        else:
                            console.print(f"  🤖 **Worker Thought:** {msg.get('content', '')}", style="green")

                elif node_name == "verify_subtask":
                    if "completed_steps" in state_update:
                        console.print(f"  ✅ **Reflection:** {state_update['completed_steps'][-1]}", style="bold blue")
                    else:
                        console.print(f"  🔄 **Reflection [Retry]:** Need more work...", style="dim white")
                        
                elif node_name == "generate_report":
                    console.print("\n🎉 **Final Execution Report:**", style="bold cyan")
                    console.print(Markdown(state_update.get("final_report", "")))

    except Exception as e:
        import traceback
        console.print(f"\n❌ **Fatal Error executing CLI Agent:** {e}", style="bold red")
        console.print(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: python scripts/hub_cli.py \"<your instruction here>\"")
        sys.exit(1)
        
    user_prompt = " ".join(sys.argv[1:])
    
    # Silence httpx and internal logging for a cleaner CLI output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("hub_gateway").setLevel(logging.WARNING)
    
    asyncio.run(run_cli_agent(user_prompt))
