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

A real-world Reinforcement Learning (RL) environment built on the OpenEnv spec. This environment simulates a Level 1 Customer Support Agent workflow, requiring an AI model to use tools to triage, diagnose, and resolve customer tickets while utilizing strict Chain of Thought (CoT) reasoning.

## Benchmark Validation

This environment has been cross-validated using proprietary, open-source, and frontier models. It successfully filters for both logical reasoning and strict tool adherence across different reasoning tiers. 

*Note: To ensure a baseline of agentic behavior, `tool_choice` is dynamically set to `"required"` for all non-Gemini models.*

| Model | Tool Choice | Avg Reward | Tier | Performance Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Gemini 3.1 Flash-Lite** | `auto` | **0.80** | SOTA Proprietary | 100% logic and SOP adherence. Flawless Chain of Thought. |
| **Qwen 2.5 7B Instruct** | `required` | **0.55** | SOTA Open-Source | Failed complex routing; escalated instead of resolving duplicate charges. |
| **Llama 3.3 70B** | `required` | **0.45** | Frontier Open | **RLHF Efficiency Bias:** Skipped mandatory policy checks entirely. |
| **Llama 3.1 8B** | `required` | **0.44** | Mid-Tier Baseline | Hallucinated KB queries, trapped in loops, failed basic task resolution. |

---

## Comparative Analysis

### 1. The 0.80 Ceiling (Gemini 3.1 Flash-Lite)
Gemini was the only model to successfully navigate the entire gauntlet. Its success was driven by its ability to utilize the `thought` parameter effectively. For example, in Task 3 (The Suspension Trap), Gemini's internal thought explicitly connected the 500 error to the `suspended` status found in billing, leading it to search the exact correct policy and escalate to `SECURITY`. 

### 2. Multi-Hop Fragility (Qwen 2.5 7B)
While Qwen is a highly capable model, it suffered a significant drop in score (0.55) due to multi-hop fragility:
* **Task 2 (Duplicate Charge):** The model correctly found the duplicate charge but defaulted to its base instinct of escalating billing issues (`escalate_ticket`) instead of following the Knowledge Base policy to issue an immediate refund (`resolve_ticket`).
* **Task 3 (Suspension):** It successfully identified that the user was suspended, but fell back on the heuristic that "500 errors go to Engineering" instead of routing to the mandated Security department.

### 3. RLHF Efficiency Bias & The "Lazy Agent" (Llama 3.3 70B)
Testing the frontier-class Llama 3.3 70B revealed a fascinating "Overconfidence Penalty," resulting in a score of 0.45:
* **Shortcutting SOP:** Even with tools required, the model completely skipped the mandatory `search_knowledge_base` tool across almost all tasks. It attempted to resolve or escalate tickets immediately based on its internal pre-trained knowledge of general support rules.
* **Observation:** This proves the environment is robust enough to catch "helpful but unaligned" behavior in massive models that prioritize speed over strict protocol adherence.

### 4. Mid-Tier Looping (Llama 3.1 8B)
Llama 3.1 8B (0.44) demonstrated the limits of smaller models in strict environments. In Task 2, it scored a flat **0.00** by failing to comprehend the duplicate charge and incorrectly escalating. In Task 4, it got trapped in a repetitive loop, repeatedly querying the Knowledge Base for "escalation path" and finding nothing, rather than synthesizing the conflicting policies.

---

## Real-World Utility

Customer support triage is a massive bottleneck for modern SaaS companies. This environment provides a realistic training ground for LLM agents to learn how to:
1. **Cross-reference user claims** against internal live databases (Billing).
2. **Search and apply strict company policies** from a central Knowledge Base.
3. **Determine Resolution Path:** Decide when to resolve a ticket autonomously vs. when to escalate to human specialists (Billing, Engineering, Security).

---

## Action & Observation Spaces

All interactions happen through strictly typed MCP Tools (Action Space). **Crucially, every tool requires a mandatory `thought` parameter** to force step-by-step reasoning before execution:
* `read_ticket(thought)`: Returns the customer's issue text. (ALWAYS START HERE).
* `search_knowledge_base(thought, query)`: Returns policy text based on semantic keywords.
* `check_billing(thought, user_id)`: Returns recent transactions and account status.
* `escalate_ticket(thought, department)`: Ends the episode, routing to a human team (Billing, Engineering, Security).
* `resolve_ticket(thought, message)`: Ends the episode with a direct customer response.

**Observation Space:** Returns JSON strings containing the tool execution results, current step count, and partial trajectory rewards. This allows the model to "see" the output of its actions in real-time.

---

## Task Difficulty & Grading Logic

The environment dynamically cycles through 4 tasks of increasing difficulty upon calling `/reset`.

| Task | Difficulty | Objective | Correct Action |
|------|------------|-----------|----------------|
| **1. Password Reset** | Easy | User forgot password. No DB check needed. | `resolve_ticket` with specific keywords (link, spam, etc.) |
| **2. Duplicate Charge** | Medium | User reports a double charge on their statement. | `resolve_ticket` directly after verifying array data. |
| **3. Suspension Diagnosis** | Hard | User reports a 500 error; DB shows suspended status. | `escalate_ticket` to **Security**. |
| **4. Cross-Policy Conflict**| Trap | User requests account deletion AND a refund. | `escalate_ticket` to **Billing** (Policy Conflict). |

**Reward Function:**
The environment provides dense, partial trajectory rewards to guide the agent:
* **+0.1 to +0.2:** For starting by reading the ticket and using the correct initial tool sequence.
* **+0.3 to +0.5:** For selecting the correct terminal action (Resolve vs. Escalate) and the correct department.
* **-0.2 to -0.5 (Penalty):** For hallucinating a resolution that violates searched policies, skipping mandatory data checks, or missing cross-policy conflicts.

---

## Robustness Testing
To ensure the environment successfully filters for intelligence, we tested models utilizing a forced Chain of Thought (`thought` parameter). Because `tool_choice` is dynamically set to `"required"` for open-source and API models, the benchmark isolates logical routing from basic tool-calling syntax errors. As demonstrated by the scoring spread, models lacking native multi-hop reasoning or strict instruction-following will consistently fail due to heuristic biases or loop-trapping. This confirms that solving the environment requires sophisticated logical branching, and a baseline reward of >0.80 requires 100% adherence to the provided SOP.

## Setup & Usage

### 1. Local Development
```bash
# Install dependencies
uv sync

# Run the FastAPI server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --env-file .env