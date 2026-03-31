---
title: Support Triage Environment
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - agents
---

# Customer Support Triage Environment

A real-world Reinforcement Learning (RL) environment built on the OpenEnv spec. This environment simulates a Level 1 Customer Support Agent workflow, requiring an AI model to use tools to triage, diagnose, and resolve customer tickets.

## Hackathon Baseline Score: 0.80

Using `gemini-3.1-flash-lite-preview` as a baseline zero-shot agent, this environment is fully solvable, achieving an average trajectory reward of 0.80 across all difficulty levels.

---

## Real-World Utility
Customer support triage is a massive bottleneck for modern SaaS companies. This environment provides a realistic training ground for LLM agents to learn how to:
1. Cross-reference user claims against internal databases.
2. Search and apply strict company policies.
3. Decide when to resolve a ticket autonomously vs. when to escalate to human specialists (Billing, Engineering, Security).

---

## Action & Observation Spaces

All interactions happen through strictly typed MCP Tools (Action Space):
* `read_ticket()`: Returns the customer's issue text.
* `search_knowledge_base(query)`: Returns policy text based on semantic keywords.
* `check_billing(user_id)`: Returns recent transactions and account status.
* `escalate_ticket(department)`: Ends the episode, routing to a human team.
* `resolve_ticket(message)`: Ends the episode with a direct customer response.

**Observation Space:** Returns JSON strings containing the tool execution results, current step count, and partial trajectory rewards.

---

## Task Difficulty & Grading Logic

The environment dynamically cycles through 4 tasks of increasing difficulty upon calling `/reset`.

| Task | Difficulty | Objective | Correct Action |
|------|------------|-----------|----------------|
| **1. Password Reset** | Easy | User forgot password. No DB check needed. | `resolve_ticket` with specific keywords (link, spam, etc.) |
| **2. Refund Request** | Medium | User wants a refund for a charge >30 days old. | `escalate_ticket` to Billing (Policy Violation). |
| **3. Server Outage** | Hard | User reports a 500 error affecting production. | `escalate_ticket` to Engineering immediately. |
| **4. Data Export** | Trap | Unsolvable with current tools. | `escalate_ticket` (Agent must recognize its own limits). |

**Reward Function:**
The environment provides dense, partial trajectory rewards:
* +0.1 for starting by reading the ticket.
* +0.1 for utilizing the correct required tools.
* +0.3 to +0.5 for selecting the correct terminal action and department.
* +0.0 to +0.2 keyword ratio matching for successful resolutions.
* -0.2 (Penalty) for hallucinating a resolution without checking required databases.

---

## Setup & Usage

### 1. Local Development
```bash
# Install dependencies
uv sync

# Run the FastAPI server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --env-file .env