# Agent Hub Vision

## The Orchestrator of the Future
The Agent Hub is designed to be the central, intelligent nervous system connecting diverse services, robust background tools, and human-facing interfaces (like Telegram via OpenClaw). It is not just a stateless router; it is a **self-evolving, stateful, agentic orchestrator** capable of understanding intent, executing complex multi-step workflows, and writing its own code to fill capability gaps.

## Core Pillars

1. **Self-Evolution via `hub_evolve`**
   The Agent Hub recognizes its own limitations. When a user requests a feature that does not exist across any connected service, the Hub seamlessly routes a specification to the `langgraph-poc` AI coding engine, automatically scaffolding, testing, and pushing new features to git branches. The system builds itself.

2. **Framework-Agnostic Discovery (MCP)**
   Tools should not be hardcoded or scraped using brittle regex. The Hub dynamically ingests standardized `openapi.json` schemas from any framework—FastAPI, Spring Boot, Go, or Rust—making the ecosystem infinitely extensible without requiring manual gateway updates.

3. **Resilient Agentic Workflows**
   Powered by **LangGraph**, the Hub breaks free from simple `while` loop chat scripts. It maintains a deterministic, graph-based execution state, allowing for human-in-the-loop approvals, continuous streaming, safe error-recovery, and deep multi-turn planning.

4. **Asynchronous Background Mastery**
   The Hub is designed for heavy lifting. By delegating long-running CLI tools (like research or compilation tasks) to asynchronous **ARQ & Redis** worker queues, the Hub remains highly responsive, simply polling task statuses natively rather than blocking main threads.

5. **Semantic Long-Term Memory**
   Every user interaction shapes the Hub's behavior. Using **Qdrant Vector Storage**, the Hub automatically extracts user preferences and directives, embedding them in a long-term memory store. When a user returns, their context is dynamically injected into the system prompt, creating a truly personalized, continuous collaboration.

6. **Ironclad Output Guardrails**
   Execution must be safe. By wrapping LLM outputs in **Instructor** and strictly enforcing **Pydantic** typing, the Hub prevents hallucinated shell injections and blocks destructive absolute paths. The agent is powerful, but securely bounded.

7. **Autonomous Multi-Agent Planning (CLI)**
   Complex goals often require intelligent subdivision. By accepting input directly via a CLI (e.g. `python scripts/hub_cli.py "make techfounder tenant..."`), the Hub can route the prompt to a **Prompt Engineer/Upgrader Agent**. This agent expands the user's brief input into a comprehensive multi-step prompt, which is then passed to a **Planner Agent** to create a structured queue of sub-tasks. The LangGraph engine then steps through these tasks procedurally—running jobs, reflecting on output, and adapting until the overarching goal is met.

## The End State
A conversational command center that requires zero manual wiring. Developers spin up new microservices, and the Agent Hub instantly discovers their capabilities, routes natural language requests to them safely, remembers user preferences effortlessly, and writes its own microservices when the existing ones fall short.
