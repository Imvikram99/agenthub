# Agent Hub

Self-evolving LLM-powered command center. Routes natural language to registered projects via tool calling.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY
python -m app.gateway --port 9000
```

## Architecture

```
projects.yaml          ← register projects here
app/
  gateway.py           ← FastAPI server + LLM conversation loop
  hub/
    discovery.py       ← auto-discovers API routes from source code
    registry.py        ← loads projects, builds tools, executes them
```

## Registering a Project

Add to `projects.yaml`:

```yaml
my_project:
  type: api
  root: /path/to/project
  url: http://localhost:8002
  description: "What this project does"
  discovery:
    method: scan_routes          # auto-scan FastAPI decorators
    entry_file: app/main.py
```

Or register at runtime:

```bash
curl -X POST http://localhost:9000/register \
  -H "Content-Type: application/json" \
  -d '{"name":"my_project","type":"api","url":"http://localhost:8002"}'
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Send natural language, get AI response |
| `/health` | GET | Check hub + backend health |
| `/projects` | GET | List registered projects |
| `/tools` | GET | List all available tools |
| `/discover` | POST | Re-scan projects for new endpoints |
| `/register` | POST | Register a new project at runtime |
