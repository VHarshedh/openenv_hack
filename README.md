---
title: Support Triage Environment
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - agents
---
# Customer Support Triage Agent (OpenEnv)

A highly robust, anti-cheat simulated OpenEnv environment designed to test and train Large Language Models on complex, multi-step customer support triage tasks.

## Motivation & Real-World Utility

Customer support routing is a critical real-world task where LLMs are frequently deployed. This environment tests an agent's ability to:

1. **Navigate structured knowledge:** Agents must dynamically query an internal Knowledge Base (KB) and a mock CRM Database.
2. **Filter Noise:** The environment procedurally generates irrelevant corporate policies (e.g., dress codes, lunch reimbursements) that agents must learn to ignore.
3. **Follow Strict Operating Procedures:** It assesses if agents blindly resolve tickets or if they securely verify account states (e.g., Suspended, Flagged) before executing escalations.
4. **RLHF & Agent Training Efficiency:** Unlike standard benchmarks with sparse, binary pass/fail grading, this environment provides a highly dense and granular reward signal (0.0 to 1.0 with intermediate penalties). This makes it an exceptionally efficient engine for generating high-quality preference pairs and trajectory data for Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

## Key Features (Anti-Cheat & Grading)

To ensure this environment rigorously challenges frontier models, it implements the following advanced features:

* **Procedural Task Generation:** Task targets and ticket text are procedurally randomized on every `reset()`. Agents cannot memorize that "USR004 is the 500 error." They must actively use their tools to extract the target user from the natural language ticket.
* **Context-Aware Thought Grading (60/40 Split):** The environment's reward function explicitly grades the agent's Chain-of-Thought (CoT). Agents receive 60% of their reward for executing the correct final tool call, but the remaining 40% is only awarded if the agent's `thought` parameter demonstrably proves it verified the KB and checked the user's DB state.
* **Anti-Teleportation & Milestone Verification:** The environment tracks a state machine of the agent's knowledge. If an agent tries to guess a dynamically generated inner-UID without explicitly querying the database first, it triggers an anti-cheat penalty.

## Task Descriptions & Difficulty

The environment contains 6 distinct tasks of escalating difficulty.

| Task | Difficulty | Objective | Expected Steps | 
| ----- | ----- | ----- | ----- | 
| 1 | **Easy** | Reset a password by verifying identity via the KB. | 3 | 
| 2 | **Medium** | Process a refund after checking the database for recent duplicate transactions. | 4 | 
| 3 | **Hard** | Deep Diagnosis: Troubleshoot an Error 500 on a Suspended account, requiring security escalation. | 4 | 
| 4 | **Trap** | Conflict Awareness: Handle a user asking for both a refund AND deletion, requiring the agent to synthesize competing KB policies. | 5 | 
| 5 | **Multi-Hop** | Shadow Ban Protocol: Agent must map an Error X77 to a secondary, hidden security protocol. | 5 | 
| 6 | **Ultra** | Project Aegis: Agent must navigate a deprecated policy, discover a hidden internal memo, find a mapped "Real UID", check a secondary Auditor status, and hit a dead-end requiring human manager ping before escalating. | 8 | 

## Model Benchmarks & Analysis

We ran inference across 5 different models to establish baselines. The strict 60/40 instruction-following grader successfully exposed significant flaws in modern tool-calling agents.

| Model | Average Score | T1 (Easy) | T2 (Med) | T3 (Hard) | T4 (Trap) | T5 (Multi) | T6 (Ultra) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Gemini 3.1 Flash-Lite** | **0.83** | 0.90 | 0.90 | 1.00 | 0.90 | 1.00 | 0.25 | 
| **Qwen 2.5 72B Instruct** | **0.70** | 0.90 | 0.25 | 1.00 | 0.90 | 1.00 | 0.15 | 
| **Qwen 2.5 7B Instruct** | **0.57** | 0.70 | 1.00 | 1.00 | 0.70 | 0.00 | 0.00 | 
| **Llama 3.1 8B Instruct** | **0.42** | 0.00 | 0.50 | 0.00 | 1.00 | 0.90 | 0.10 | 
| **Llama 3.3 70B Versatile** | **0.16** | 0.20 | 0.05 | 0.20 | 0.00 | 0.30 | 0.20 | 

### Key Insights from the Baselines

1. **SOP Adherence vs. Task Completion:** The data proves that while large models might eventually guess the correct final department, they struggle to strictly adhere to corporate Standard Operating Procedures (SOPs). Heavy penalties for bypassing the Knowledge Base or prematurely pinging human managers (-0.3 deduction) successfully differentiate true reasoning agents from lucky guessers.
2. **The Overconfidence Penalty:** `Llama-3.3-70B` scored the lowest because it was overconfident. Instead of following the SOP to search the knowledge base first, it frequently guessed the escalation department immediately based on its pre-trained weights. Our strict grader severely penalized this lack of methodology.
3. **Infinite Tool Loops:** The smaller 7B and 8B models frequently fell into infinite tool loops. For example, `Qwen 2.5 7B` failed Task 5 because it called `read_ticket` 15 times in a row. `Llama 3.3 70B` bizarrely called `check_billing` 14 consecutive times on Task 4.
4. **The Ultra Task (Project Aegis):** No model was able to fully solve Task 6. Even the top performer (Gemini) successfully navigated the multi-hop database lookup but failed to synthesize the final rule after the manager ping returned a system auto-reply.

## Action & Observation Spaces

This environment strictly implements the **OpenEnv MCP (Model Context Protocol)** specification.

### Observation Space

The observation space returns the result of the tool execution as a string, alongside the standard OpenEnv variables:

* `done` (bool): Whether the episode has terminated.
* `reward` (float): The current trajectory reward (0.0 to 1.0).
* `metadata` (dict): Includes the difficulty of the current task.

### Action Space (Tools)

The agent interacts with the environment via 6 typed tools. **Every tool strictly requires a `thought` string parameter.**

1. `read_ticket(thought)`: Reads the inbound customer message.
2. `search_knowledge_base(thought, query)`: Searches the internal policy KB.
3. `check_billing(thought, user_id)`: Checks account status and CRM notes.
4. `ping_human_manager(thought, reason)`: Requests assistance on undocumented edge cases.
5. `escalate_ticket(thought, department)`: Terminal action. Routes the ticket.
6. `resolve_ticket(thought, message)`: Terminal action. Replies directly to the customer.

## Setup & Usage

### 1. Run the Environment (Docker)

This environment is containerized and ready to deploy to Hugging Face Spaces.

```bash
docker build -t support_env .
docker run -p 8000:8000 support_env
```

### 2. Run Inference (Evaluation)

We provide a baseline inference.py script that utilizes the OpenAI API client format to run evaluations.

```bash
# Set your environment variables
export ENV_URL="http://localhost:8000"
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export MODEL_NAME="gemini-3.1-flash-lite-preview"
export HF_TOKEN="your_token_here"

# Run the evaluation
python inference.py > output.log
