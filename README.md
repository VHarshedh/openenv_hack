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

## Benchmark Validation

This environment has been cross-validated using proprietary, open-source, and frontier models. It successfully filters for both logical reasoning and strict tool adherence across different reasoning tiers.

| Model | Tool Choice | Avg Reward | Tier | Performance Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Gemini 3.1 Flash-Lite** | `auto` | **0.80** | SOTA Proprietary | 100% logic and SOP adherence. |
| **Qwen 2.5 7B Instruct** | `auto` | **0.80** | SOTA Open-Source | Identical performance to Gemini. |
| **Llama 3.1 8B** | `required`| **0.55** | Mid-Tier Baseline | Follows tools but fails complex date math. |
| **Llama 3.3 70B** | `auto` | **0.50** | Frontier Open | **RLHF Efficiency Bias:** Skipped required tools. |
| **Llama 3.1 8B** | `auto` | **0.00** | Legacy | Fails tool adherence (chatted JSON as text). |

---

## Comparative Analysis

### 1. Gemini vs. Qwen (The 0.80 Trajectories)
While both SOTA models reached the same final score, their trajectories showed subtle differences in semantic reasoning:
* **Query Precision:** Gemini 3.1 Flash-Lite demonstrated slightly higher verbosity in searches (e.g., searching for `"data export policy"`), whereas Qwen 2.5 7B optimized for keyword efficiency (`"data export"`).
* **Logic Consistency:** Both models successfully navigated the **"Trap"** (recognizing a "No Result" from the KB and escalating to Engineering) and the **"Date Math"** (correctly parsing a March 1st transaction against the system clock to trigger a Billing Escalation).

### 2. RLHF Efficiency Bias & The "Lazy Agent" (Llama 3.3 70B)
Testing the frontier-class Llama 3.3 70B (via Groq) revealed a fascinating "Overconfidence Penalty." Despite its high intelligence, the model achieved a lower score (0.50) due to **RLHF Efficiency Bias**:
* **Shortcutting SOP:** The model attempted to resolve tickets immediately based on its internal knowledge of general support rules, skipping the mandatory `search_knowledge_base` and `check_billing` steps required by the environment.
* **Tool Misalignment:** In outage tasks, the model used the `resolve_ticket` tool to *inform* the user it was escalating, rather than physically calling the `escalate_ticket` tool to move the state machine forward.
* **Observation:** This proves the environment is robust enough to catch "helpful but unaligned" behavior in massive models that prioritize speed/helpfulness over strict protocol adherence.

### 3. Llama 3.1 8B Ablation Study (The 0.55 Trajectory)
Testing Llama 3.1 8B revealed the environment's ability to expose logical gaps in mid-tier models when forced into tool compliance (`tool_choice="required"`):
* **Task 2 (Refund Logic Failure):** The model correctly searched the KB and checked billing for the user. However, it completely ignored the "30-day policy limit" and attempted to resolve the ticket with a refund anyway, triggering the environment's penalty logic.
* **Task 4 (The Trap):** When faced with a missing policy, it correctly chose to escalate but selected the wrong department (`billing` instead of `engineering`), demonstrating a lower semantic understanding of edge-case routing.

---

## Real-World Utility

Customer support triage is a massive bottleneck for modern SaaS companies. This environment provides a realistic training ground for LLM agents to learn how to:
1. **Cross-reference user claims** against internal live databases (Billing).
2. **Search and apply strict company policies** from a central Knowledge Base.
3. **Determine Resolution Path:** Decide when to resolve a ticket autonomously vs. when to escalate to human specialists (Billing, Engineering, Security).

---

## Action & Observation Spaces

All interactions happen through strictly typed MCP Tools (Action Space):
* `read_ticket()`: Returns the customer's issue text. (ALWAYS START HERE).
* `search_knowledge_base(query)`: Returns policy text based on semantic keywords.
* `check_billing(user_id)`: Returns recent transactions and account status.
* `escalate_ticket(department)`: Ends the episode, routing to a human team (Billing, Engineering, Security).
* `resolve_ticket(message)`: Ends the episode with a direct customer response.

**Observation Space:** Returns JSON strings containing the tool execution results, current step count, and partial trajectory rewards. This allows the model to "see" the output of its actions in real-time.

---

## Task Difficulty & Grading Logic

The environment dynamically cycles through 4 tasks of increasing difficulty upon calling `/reset`.

| Task | Difficulty | Objective | Correct Action |
|------|------------|-----------|----------------|
| **1. Password Reset** | Easy | User forgot password. No DB check needed. | `resolve_ticket` with specific keywords (link, spam, etc.) |
| **2. Refund Request** | Medium | User wants a refund for a charge >30 days old. | `escalate_ticket` to Billing (Policy Violation). |
| **3. Server Outage** | Hard | User reports a 500 error affecting production. | `escalate_ticket` to Engineering immediately. |
| **4. Data Export** | Trap | Policy doesn't exist for the requested data type. | `escalate_ticket` (Agent must recognize its own limits). |

**Reward Function:**
The environment provides dense, partial trajectory rewards to guide the agent:
* **+0.1 to +0.2:** For starting by reading the ticket and using the correct initial tool sequence.
* **+0.3 to +0.5:** For selecting the correct terminal action (Resolve vs. Escalate) and the correct department.
* **-0.2 to -0.5 (Penalty):** For hallucinating a resolution that violates searched policies or skipping mandatory data checks.

---

## Robustness Testing
To ensure the environment successfully filters for intelligence, we tested "Low-Reasoning" models without forced tool compliance. As demonstrated by the Llama 3.1 8B (`auto`) score of **0.00**, models lacking native agentic structure will consistently fail due to loop-trapping. This confirms that solving the environment requires sophisticated logical branching, strict JSON adherence, and a baseline reward of 0.80 requires 100% adherence to the provided SOP.

## Setup & Usage

### 1. Local Development
```bash
# Install dependencies
uv sync

# Run the FastAPI server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --env-file .env