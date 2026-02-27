import json
import logging
import os
from typing import Annotated, Any, Dict, List, Optional
import operator

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from app.gateway import (
    HUB_MODEL,
    client, 
    raw_client,
    get_system_prompt, 
    get_all_tools, 
    execute_tool, 
    logger,
    validate_tool_arguments
)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_LLM_CALLS_PER_TASK = int(os.environ.get("CLI_MAX_LLM_CALLS", "30"))
MAX_SUBTASK_RETRIES = 4

# ── Pydantic Models for Planning ──────────────────────────────────────────────

class SubTask(BaseModel):
    id: int = Field(..., description="Sequential step number (1, 2, 3...)")
    description: str = Field(..., description="Detailed description of the operation to perform")
    expected_outcome: str = Field(..., description="What indicates this step is complete and successful?")

class ExecutionPlan(BaseModel):
    subtasks: List[SubTask] = Field(..., description="List of sequential subtasks to complete the main objective")

# ── Graph State ──────────────────────────────────────────────────────────────

def custom_messages_reducer(left: list, right: list | str | dict) -> list:
    """Custom reducer that preserves purely explicit Python dicts. If purely 'CLEAR', erases."""
    if right == "CLEAR":
        return []
    if not left:
        left = []
    if isinstance(right, list):
        return left + right
    return left + [right]

class CLIState(TypedDict):
    original_prompt: str
    upgraded_prompt: str
    plan: List[SubTask]
    current_task_index: int
    worker_messages: Annotated[list, custom_messages_reducer]
    completed_steps: Annotated[list, operator.add]
    final_report: str
    retry_count: int
    llm_call_count: int  # Track total LLM calls to enforce cost cap

# ── Nodes ────────────────────────────────────────────────────────────────────

async def upgrade_prompt_node(state: CLIState) -> Dict[str, Any]:
    """Expands the user's brief input into a wealthy system prompt."""
    logger.info("Nodes -> Upgrading Prompt")
    original = state.get("original_prompt", "")
    
    sys_prompt = get_system_prompt()
    available_tools = get_all_tools()
    tools_list = "\n".join([f"- {t['function']['name']}: {t['function'].get('description','')}" for t in available_tools])
    
    prompt = f"""
You are an expert autonomous Agent Orchestrator. The user has provided a brief CLI directive:
"{original}"

Your task is to rewrite this into a highly detailed, extremely explicit set of instructions.
We have the following tools available:
{tools_list}

Write a comprehensive prompt that explains exactly what needs to be achieved, potential edge cases to look out for, and which logic / tools are likely needed.
"""
    
    # We just use the standard completions call for a text response
    response = await raw_client.chat.completions.create(
        model=HUB_MODEL,
        messages=[
            {"role": "system", "content": "You are a prompt engineering AI."},
            {"role": "user", "content": prompt}
        ],
    )
    
    upgraded = response.choices[0].message.content or original
    logger.info(f"Upgraded Prompt length: {len(upgraded)} characters")
    return {"upgraded_prompt": upgraded}

async def create_plan_node(state: CLIState) -> Dict[str, Any]:
    """Generates an explicit list of subtasks using Instructor."""
    logger.info("Nodes -> Creating Execution Plan")
    upgraded = state.get("upgraded_prompt", "")
    
    prompt = f"""
Based on the following comprehensive instructions, break down the execution into precise, sequential subtasks.
Avoid grouping unrelated API calls together if they depend on earlier reflections.

Instructions:
{upgraded}
"""
    try:
        plan: ExecutionPlan = await client.chat.completions.create(
            model=HUB_MODEL,
            response_model=ExecutionPlan,
            messages=[
                {"role": "system", "content": "You are a logical task planner. Break the objective down into discrete subtasks."},
                {"role": "user", "content": prompt}
            ],
            max_retries=3
        )
        logger.info(f"Created plan with {len(plan.subtasks)} subtasks.")
        
        # Init state variables for the worker loop
        return {
            "plan": plan.subtasks,
            "current_task_index": 0,
            "completed_steps": [],
            "worker_messages": [],
            "retry_count": 0
        }
    except Exception as e:
        logger.error(f"Failed to create plan: {e}")
        # Fallback to a single generic subtask
        fallback = SubTask(id=1, description=upgraded, expected_outcome="Task finished.")
        return {
            "plan": [fallback],
            "current_task_index": 0,
            "completed_steps": [],
            "worker_messages": [],
            "retry_count": 0
        }

async def run_subtask_node(state: CLIState) -> Dict[str, Any]:
    """The Worker: Focuses solely on completing the current subtask."""
    plan = state.get("plan", [])
    idx = state.get("current_task_index", 0)
    
    if idx >= len(plan):
        return {}
    
    # Enforce max LLM calls to prevent runaway costs
    call_count = state.get("llm_call_count", 0)
    if call_count >= MAX_LLM_CALLS_PER_TASK:
        logger.warning(f"Hit max LLM call limit ({MAX_LLM_CALLS_PER_TASK})")
        return {"worker_messages": [{"role": "assistant", "content": f"Aborted: hit max LLM call limit ({MAX_LLM_CALLS_PER_TASK}). Forcing completion."}]}

    current_task = plan[idx]
    logger.info(f"Nodes -> Running Subtask {idx+1}/{len(plan)}: {current_task.description[:60]}...")
    
    # Build context from previous steps
    completed = state.get("completed_steps", [])
    history_ctx = "\n".join(completed) if completed else "None."
    
    messages = state.get("worker_messages", [])
    
    if not messages:
        # First iteration of this subtask. Initialize the context.
        focus_prompt = f"""
{get_system_prompt()}

# Overall Context
You are currently on Step {idx+1} of {len(plan)}.
Previously completed steps summary:
{history_ctx}

# CURRENT TASK
Your sole focus right now is to accomplish the following task:
Description: {current_task.description}
Expected Outcome: {current_task.expected_outcome}

Execute tools necessary to fulfill this specific step.
If you have completed the step, summarize your findings in a final text response.
"""
        messages = [{"role": "system", "content": focus_prompt}]
        
    tools = get_all_tools()
    
    # Format for raw OpenAI usage to support tool calling correctly
    payload = {
        "model": HUB_MODEL,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        
    try:
        # Standard OpenAI call
        response = await raw_client.chat.completions.create(**payload)
        msg_obj = response.choices[0].message
        
        message_dict = {"role": msg_obj.role, "content": msg_obj.content or ""}
        
        if msg_obj.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments if tc.function.arguments and tc.function.arguments.strip() else "{}",
                    }
                }
                for tc in msg_obj.tool_calls
            ]
            
        return {"worker_messages": [message_dict], "llm_call_count": call_count + 1}
    except Exception as e:
        logger.error(f"Worker LLM error: {e}")
        return {"worker_messages": [{"role": "assistant", "content": f"Error: {e}"}], "llm_call_count": call_count + 1}


async def execute_tools_node(state: CLIState) -> Dict[str, Any]:
    """Executes the tools identified by the worker LLM in the current subtask."""
    messages = state.get("worker_messages", [])
    if not messages:
        return {}
        
    last_message = messages[-1]
    tool_calls = last_message.get("tool_calls", [])
    
    results = []
    for tc in tool_calls:
        fn = tc["function"]
        tool_name = fn["name"]
        try:
            args_str = fn.get("arguments", "{}")
            if not args_str.strip():
                args_str = "{}"
            tool_args = json.loads(args_str)
            tool_args = validate_tool_arguments(tool_name, tool_args)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON args for tool {tool_name}: '{args_str}'")
            tool_args = {}
        except ValueError as ve:
            logger.warning(f"Validation blocked CLI subtask tool call: {ve}")
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tool_name,
                "content": json.dumps({"error": f"Guardrail: {ve}"})
            })
            continue

        logger.info(f"Subtask Tool Exec: {tool_name}({str(tool_args)[:100]}...)")
        result = await execute_tool(tool_name, tool_args)
        
        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tool_name,
            "content": result
        })
        
    return {"worker_messages": results}


class SubtaskReflection(BaseModel):
    is_complete: bool = Field(..., description="Is the current subtask completely finished based on the expected outcome?")
    summary: str = Field(..., description="A 1-2 sentence summary of what was actually achieved or the final blocker.")


async def generate_final_report_node(state: CLIState) -> Dict[str, Any]:
    """Generates the final wrap-up report after all tasks finish."""
    completed = state.get("completed_steps", [])
    report = "### Multi-Agent Execution Complete\n\n"
    for step in completed:
        report += f"- {step}\n"
    
    logger.info("Nodes -> Final Report Generated")
    return {"final_report": report}

# ── Routing Logic ────────────────────────────────────────────────────────────

def route_worker(state: CLIState) -> str:
    messages = state.get("worker_messages", [])
    last_message = messages[-1] if messages else {}
    if last_message.get("tool_calls"):
        return "execute_tools"
    return "verify_subtask"

def route_planner(state: CLIState) -> str:
    plan = state.get("plan", [])
    idx = state.get("current_task_index", 0)
    if idx >= len(plan):
        return "generate_report"
    return "run_subtask"


async def verify_subtask_node(state: CLIState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    idx = state.get("current_task_index", 0)
    current_task = plan[idx]
    messages = state.get("worker_messages", [])
    
    logger.info("Nodes -> Verifying Subtask Completion")
    
    history_dump = "\n".join([f"{m['role'].upper()}: {m.get('content', 'tool_calls')}" for m in messages[-10:]])
    prompt = f"TASK: {current_task.description}\nOUTCOME: {current_task.expected_outcome}\nHISTORY:\n{history_dump}\nComplete?"
    
    try:
        reflection: SubtaskReflection = await client.chat.completions.create(
            model=HUB_MODEL,
            response_model=SubtaskReflection,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Is the task completely finished?"},
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        reflection = SubtaskReflection(is_complete=True, summary="Assumed true due to error.")

    if reflection.is_complete or state.get("retry_count", 0) > 4:
        summary_log = f"Step {idx+1}: {reflection.summary}"
        return {
            "current_task_index": idx + 1,
            "completed_steps": [summary_log],
            "worker_messages": "CLEAR", # the custom reducer intercepts this
            "retry_count": 0
        }
    else:
        return {
            "worker_messages": [{"role": "user", "content": f"Evaluator: Not done. {reflection.summary}"}],
            "retry_count": state.get("retry_count", 0) + 1
        }


# ── Build the Graph ─────────────────────────────────────────────────────────

cli_workflow = StateGraph(CLIState)

cli_workflow.add_node("upgrade_prompt", upgrade_prompt_node)
cli_workflow.add_node("create_plan", create_plan_node)
cli_workflow.add_node("run_subtask", run_subtask_node)
cli_workflow.add_node("execute_tools", execute_tools_node)
cli_workflow.add_node("verify_subtask", verify_subtask_node)
cli_workflow.add_node("generate_report", generate_final_report_node)

cli_workflow.add_edge(START, "upgrade_prompt")
cli_workflow.add_edge("upgrade_prompt", "create_plan")
cli_workflow.add_edge("create_plan", "run_subtask")

cli_workflow.add_conditional_edges("run_subtask", route_worker)
cli_workflow.add_edge("execute_tools", "run_subtask")

cli_workflow.add_conditional_edges("verify_subtask", route_planner, {
    "run_subtask": "run_subtask",
    "generate_report": "generate_report"
})
cli_workflow.add_edge("generate_report", END)

cli_graph = cli_workflow.compile()
