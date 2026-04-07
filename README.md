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

# 🛡️ ComplianceGuard: Support Triage Environment

A highly robust, anti-cheat simulated OpenEnv environment designed to test and train Large Language Models on complex, multi-step customer support triage tasks.

## Motivation & Real-World Utility

Customer support routing is a critical real-world task where LLMs are frequently deployed. This environment tests an agent's ability to:

* **Navigate structured knowledge:** Agents must dynamically query an internal Knowledge Base (KB) and a mock CRM Database.
* **Filter Noise:** The environment procedurally generates irrelevant corporate policies (e.g., dress codes, lunch reimbursements) that agents must learn to ignore.
* **Follow Strict Operating Procedures:** It assesses if agents blindly resolve tickets or if they securely verify account states (e.g., Suspended, Flagged) before executing escalations.
* **RLHF & Agent Training Efficiency:** Unlike standard benchmarks with sparse, binary pass/fail grading, this environment provides a highly dense and granular reward signal (0.01 to 0.99 with intermediate penalties). This makes it an exceptionally efficient engine for generating high-quality preference pairs and trajectory data for Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

## Key Features (Anti-Cheat & Grading)

To ensure this environment rigorously challenges frontier models, it implements the following advanced features:

* **Procedural Task Generation:** Task targets and ticket text are procedurally randomized on every `reset()`. Agents cannot memorize that "USR004 is the 500 error." They must actively use their tools to extract the target user from the natural language ticket.
* **Context-Aware Thought Grading (60/40 Split):** The environment's reward function explicitly grades the agent's Chain-of-Thought (CoT). Agents receive 60% of their reward for executing the correct final tool call, but the remaining 40% is only awarded if the agent's thought parameter demonstrably proves it verified the KB and checked the user's DB state.
* **Anti-Teleportation & Milestone Verification:** The environment tracks a state machine of the agent's knowledge. If an agent tries to guess a dynamically generated inner-UID without explicitly querying the database first, it triggers an anti-cheat penalty.

## Task Descriptions & Difficulty

The environment evaluates agents across a dynamic 10-task curriculum, scaling in difficulty:

| Task | Difficulty | Objective | Expected Steps |
|---|---|---|---|
| 1 | Easy | **Password Reset:** Reset a password by verifying identity via the KB. | 3 |
| 2 | Medium | **Duplicate Charge:** Process a refund after checking the database for recent duplicate transactions. | 4 |
| 3 | Hard | **Deep Diagnosis:** Troubleshoot an Error 500 on a Suspended account, requiring security escalation. | 4 |
| 4 | Trap | **Conflict Awareness:** Handle a user asking for both a refund AND account deletion. Agent must search both policies and escalate to billing. | 5 |
| 5 | VIP | **VIP Payment Update:** Agent must detect a VIP flag in the CRM and route to billing instead of resolving directly. | 5 |
| 6 | Multi-Hop | **Shadow Ban Protocol:** Agent must map an Error X77 to a secondary, hidden security protocol. | 5 |
| 7 | Breach | **Security Breach Report:** Follow a 2-hop DB + KB chain to verify a Security Auditor's status. | 6 |
| 8 | Privacy | **GDPR Hold:** Strict word-boundary policy matching and Privacy Officer routing. | 7 |
| 9 | Ultra | **Project Aegis:** Navigate a deprecated policy chain, resolve a Real UID alias, and verify a Compliance Auditor via a 3-hop database sequence. | 8 |
| 10 | Mega | **Mega Chain:** Triple-flag account (X77 + compliance hold + shadow ban). Synthesize intersecting policies and execute strict SLA escalation. | 10 |

## Model Benchmarks & Analysis

We ran inference across 5 different models to establish baselines. The strict 60/40 instruction-following grader successfully exposed significant flaws in modern tool-calling agents.

| Model | Average Score | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Gemini 3.1 Flash-Lite** | ~0.78 | 0.90 | 0.74 | 0.99 | 0.41 | 0.99 | 0.99 | 0.85 | 0.82 | 0.49 | 0.59 |
| **Qwen 2.5 72B Instruct** | ~0.76 | 0.89 | 0.96 | 0.99 | 0.55 | 0.96 | 0.98 | 0.37 | 0.88 | 0.36 | 0.62 |
| **Qwen 2.5 7B Instruct** | ~0.61 | 0.85 | 0.84 | 0.95 | 0.50 | 0.79 | 0.82 | 0.43 | 0.40 | 0.35 | 0.13 |
| **Llama 3.1 8B Instruct** | ~0.65 | 0.36 | 0.65 | 0.99 | 0.55 | 0.99 | 0.91 | 0.57 | 0.54 | 0.50 | 0.44 |
| **Llama 3.3 70B Versatile** | ~0.26 | 0.75 | 0.19 | 0.73 | 0.16 | 0.22 | 0.12 | 0.11 | 0.17 | 0.09 | 0.01 |

🏆 **Best Performer: Gemini 3.1 Flash-Lite**
Gemini 3.1 Flash-Lite Preview achieved the highest performance across all evaluations. The highest single evaluation run was achieved by **Gemini 3.1 Flash-Lite** with a remarkable score of **~0.81** (0.810). It perfectly navigated the strict HTTP concurrency state-tracking and proved highly resilient to the environment's `SYSTEM_REJECT` soft-blocks. Unlike other models that collapsed into infinite tool loops when encountering a hard-stop, Gemini successfully synthesized the environment's feedback, corrected its SOP sequence, and achieved a near-perfect trajectory up through Task 6.

### Key Insights from the Baselines

* **SOP Adherence vs. Task Completion:** The data proves that while large models might eventually guess the correct final department, they struggle to strictly adhere to corporate Standard Operating Procedures (SOPs). Heavy penalties for bypassing the Knowledge Base or prematurely pinging human managers (-0.3 deduction) successfully differentiate true reasoning agents from lucky guessers.
* **The Overconfidence Penalty:** Llama-3.3-70B scored the lowest because it was overconfident. Instead of following the SOP to search the knowledge base first, it frequently guessed the escalation department immediately based on its pre-trained weights. Our strict grader severely penalized this lack of methodology via `SYSTEM_REJECT` blocks.
* **Infinite Tool Loops:** The smaller 7B and 8B models frequently fell into infinite tool loops. For example, Qwen 2.5 7B failed late-stage tasks by calling `read_ticket` 15 times in a row. Llama 3.3 70B bizarrely called `check_billing` 14 consecutive times on Task 4.
* **The Ultra & Mega Tasks:** No zero-shot model was able to fully solve Tasks 9 and 10. Even the top performer (Gemini) successfully navigated the multi-hop database lookup but eventually tripped the dynamic efficiency SLA penalties (-0.05 per extra step) while attempting to synthesize the intersecting compliance rules.

## Action & Observation Spaces

This environment strictly implements the OpenEnv MCP (Model Context Protocol) specification.

### Observation Space

The observation space returns the result of the tool execution as a string, alongside the standard OpenEnv variables:

* `done` (bool): Whether the episode has terminated.
* `reward` (float): The current trajectory reward (clamped safely between 0.01 and 0.99).
* `metadata` (dict): Includes the difficulty of the current task.

### Action Space (Tools)

The agent interacts with the environment via 6 typed tools. Every tool strictly requires a `thought` string parameter.

* `read_ticket(thought)`: Reads the inbound customer message.
* `search_knowledge_base(thought, query)`: Searches the internal policy KB.
* `check_billing(thought, user_id)`: Checks account status and CRM notes.
* `ping_human_manager(thought, reason)`: Decoy trap. Always triggers a −0.3 reward penalty regardless of task. Agents must use the Knowledge Base and database to resolve every scenario without human intervention.
* `escalate_ticket(thought, department)`: Terminal action. Routes the ticket.
* `resolve_ticket(thought, message)`: Terminal action. Replies directly to the customer.

## Setup & Usage

### 1. Run the Environment (Docker)

This environment is containerized and ready to deploy to Hugging Face Spaces.

```bash
docker build -t support_env .
docker run -p 8000:8000 support_env
```

### 2. Run Inference (Evaluation)

We provide a baseline `inference.py` script that utilizes the OpenAI API client format to run evaluations.

```bash
# Set your environment variables in a .env file
export ENV_URL="http://localhost:8000"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="gemini-3.1-flash-lite-preview"
export HF_TOKEN="your_token_here"

# Run the evaluation
python inference.py > output.log
```

### 3. Visualizing Results (Streamlit Dashboard)

Terminal logs can be difficult to parse. We provide a Streamlit dashboard to visually replay agent trajectories, displaying step-by-step Chain-of-Thought, tool arguments, and highlighted `SYSTEM_REJECT` process supervision blocks.

```bash
pip install streamlit pandas
streamlit run visualizer.py
```

Simply upload any `.json` file from the `results/` directory into the UI. If inference.py hasn't been run yet, you can use the sample_trajectory.json in the root directory for the visualization.

### 4. The Data Engine (Exporting DPO)

ComplianceGuard is designed to double as a training data generator.

*   **`export_dpo.py`**: Parses evaluation logs and converts them into a Hugging Face Direct Preference Optimization (DPO) dataset. Trajectories scoring >= 0.8 are labeled as **Chosen**, while those hitting a `SYSTEM_REJECT` are labeled as **Rejected**.
*   **`dpo_dataset.jsonl`**: The exported output file, ready for fine-tuning Llama 4 or Qwen models for enterprise safety.

```bash
python export_dpo.py --log_dir ./results
```

## 🧪 Project Structure & Test Suite

To ensure mathematical precision and strict API compliance, the repository includes a comprehensive test suite:

*   **`test_ws.py`**: The Gold-Standard Integration Test. Bypasses LLM latency to execute the hard-coded perfect trajectory, proving the environment is solvable and mathematically achieves a perfect 0.99 across all 10 tasks.
*   **`test_api.py`**: Validates the OpenEnv / FastMCP OpenAI client proxy integration.
*   **`test_local_env.py`**: Tests core environment logic locally without FastAPI/HTTP overhead.
*   **`test_http_reset.py`**: Validates the Phase 1 Strict Concurrency architecture, ensuring `/reset` and `/state` properly track and isolate `episode_id` lifecycles.
