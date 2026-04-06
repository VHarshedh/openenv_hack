"""
Integration test for support_env — verifies a perfect agent scores 1.0 across all 10 tasks.

Runs all tasks dynamically: UIDs are extracted fresh from each ticket/DB response
so the test is immune to the randomized database generation.

Usage:
    python test_ws.py
"""
import sys
import os
import re
import subprocess
import time
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.dirname(__file__))

from client import SupportEnv
from openenv.core.env_server.mcp_types import CallToolAction

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# ─── Helpers ────────────────────────────────────────────────────────────────

def get_text(step_result) -> str:
    """Extract plain text from a step result."""
    obs = step_result.observation
    result = getattr(obs, "result", None)
    if result is None:
        return ""
    if hasattr(result, "data"):
        return str(result.data)
    if isinstance(result, dict):
        if "data" in result:
            return str(result["data"])
        if "content" in result:
            c = result["content"]
            if isinstance(c, list) and c:
                return c[0].get("text", str(c[0])) if isinstance(c[0], dict) else str(c[0])
    return str(result)


def extract_uid(text: str) -> str:
    """Pull first USR### token out of a string."""
    m = re.search(r"USR\d{3}", text)
    return m.group(0) if m else ""


def extract_all_uids(text: str) -> list:
    """Pull all USR### tokens out of a string."""
    return re.findall(r"USR\d{3}", text)


def step(env, tool: str, **kwargs):
    return env.step(CallToolAction(tool_name=tool, arguments=kwargs))


# ─── Rich thought strings that trigger all three _validate_thought tiers ────

T_READ    = "The user's ticket inquiry requests a specific issue that I need to read carefully."
T_DB      = "The user's account status is active/suspended per the DB state I just queried."
T_KB      = "Per the policy/protocol in the knowledge base I need to verify the SOP rule."
T_FINAL   = "After verifying the KB rules and account state, I can now escalate per protocol."

# ─── Task runners ─────────────────────────────────────────────────────────

def task_1_easy(env) -> tuple:
    """Password Reset — 3 steps."""
    r = env.reset(options={"task_idx": 0})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    step(env, "search_knowledge_base",
         thought=T_KB + " password reset policy applies here.",
         query="password reset")

    # Resolve message must contain: password, reset, link or email
    sr = step(env, "resolve_ticket",
              thought=T_KB + " policy verified. Sending password reset link.",
              message=f"Hi, I have sent a password reset link to your registered email. "
                      f"Please check your inbox and spam folder. Account: {uid}.")
    return ("Task 1 (easy)", sr.reward, sr.done)


def task_2_medium(env) -> tuple:
    """Duplicate Charge Refund — 4 steps."""
    r = env.reset(options={"task_idx": 1})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    step(env, "search_knowledge_base",
         thought=T_KB + " refund policy for duplicate charge.",
         query="duplicate charge")

    db_obs = step(env, "check_billing",
                  thought="Checking user account status in database to verify duplicate charge.",
                  user_id=uid)
    db_text = get_text(db_obs)

    # Resolve message must contain: refund, process, duplicate, charge
    sr = step(env, "resolve_ticket",
              thought=T_DB + " Status verified. Processing duplicate charge refund per policy.",
              message=f"I have verified the duplicate charge on your account {uid} and have "
                      f"processed a full refund. The duplicate amount will be returned within 3-5 business days.")
    return ("Task 2 (medium)", sr.reward, sr.done)


def task_3_hard(env) -> tuple:
    """Error 500 / Suspended Account — 4 steps."""
    r = env.reset(options={"task_idx": 2})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking user account status is suspended in the DB.",
                  user_id=uid)

    step(env, "search_knowledge_base",
         thought=T_KB + " suspended account error 500 policy.",
         query="suspended account error 500")

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " Account is suspended causing 500 error. Protocol says escalate to security.",
              department="security")
    return ("Task 3 (hard)", sr.reward, sr.done)


def task_4_trap(env) -> tuple:
    """Policy Conflict — Refund + Delete — 5 steps."""
    r = env.reset(options={"task_idx": 3})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking account status and transaction date in the database.",
                  user_id=uid)

    step(env, "search_knowledge_base",
         thought=T_KB + " refund policy needs verification.",
         query="refund policy")

    step(env, "search_knowledge_base",
         thought=T_KB + " delete account policy also needs verification — conflict exists.",
         query="delete account conflict")

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " Refund and deletion conflict policy requires billing escalation.",
              department="billing")
    return ("Task 4 (trap)", sr.reward, sr.done)


def task_5_vip(env) -> tuple:
    """VIP Payment Update — 5 steps."""
    r = env.reset(options={"task_idx": 4})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking account status — looking for vip_flag in the database.",
                  user_id=uid)

    step(env, "search_knowledge_base",
         thought=T_KB + " payment method update policy.",
         query="payment method update")

    step(env, "search_knowledge_base",
         thought=T_KB + " VIP account policy protocol requires escalation not direct resolution.",
         query="vip account policy")

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " VIP account flag active — per protocol escalate to billing.",
              department="billing")
    return ("Task 5 (vip)", sr.reward, sr.done)


def task_6_multi_hop(env) -> tuple:
    """Error X77 / Shadow Ban — 5 steps."""
    r = env.reset(options={"task_idx": 5})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    step(env, "search_knowledge_base",
         thought=T_KB + " looking up X77 error code policy.",
         query="x77")

    db_obs = step(env, "check_billing",
                  thought="Checking account status and x77_flag state in the database.",
                  user_id=uid)

    step(env, "search_knowledge_base",
         thought=T_KB + " shadow ban protocol is the next rule to verify.",
         query="shadow ban protocol")

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " X77 + shadow ban protocol confirmed — escalate to security.",
              department="security")
    return ("Task 6 (multi_hop)", sr.reward, sr.done)


def task_7_breach(env) -> tuple:
    """Security Breach — 6 steps."""
    r = env.reset(options={"task_idx": 6})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking account status and security_incident flag in the database.",
                  user_id=uid)
    db_text = get_text(db_obs)

    # Extract auditor UID from the DB response
    auditor_uid = ""
    m = re.search(r"Security Auditor:\s*(USR\d{3})", db_text)
    if m:
        auditor_uid = m.group(1)

    step(env, "search_knowledge_base",
         thought=T_KB + " unauthorized login policy needs to be looked up first.",
         query="unauthorized login")

    step(env, "search_knowledge_base",
         thought=T_KB + " data breach protocol is the SOP for this incident.",
         query="data breach protocol")

    if auditor_uid:
        step(env, "check_billing",
             thought="Checking security auditor account status in the database.",
             user_id=auditor_uid)

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " Security breach with active auditor — escalate to engineering per protocol.",
              department="engineering")
    return ("Task 7 (breach)", sr.reward, sr.done)


def task_8_privacy(env) -> tuple:
    """GDPR Privacy Hold — 7 steps."""
    r = env.reset(options={"task_idx": 7})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking account status and gdpr_flag in the database.",
                  user_id=uid)
    db_text = get_text(db_obs)

    # Extract privacy officer UID from DB response
    officer_uid = ""
    m = re.search(r"Privacy Officer:\s*(USR\d{3})", db_text)
    if m:
        officer_uid = m.group(1)

    step(env, "search_knowledge_base",
         thought=T_KB + " delete account policy — checking if deletion is allowed.",
         query="delete account")

    step(env, "search_knowledge_base",
         thought=T_KB + " GDPR deletion request policy to verify proper procedure.",
         query="gdpr deletion request")

    step(env, "search_knowledge_base",
         thought=T_KB + " privacy officer routing protocol must be followed.",
         query="privacy officer")

    if officer_uid:
        step(env, "check_billing",
             thought="Checking privacy officer account status in the database.",
             user_id=officer_uid)

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " Privacy officer is red-flag — GDPR hold requires security escalation per protocol.",
              department="security")
    return ("Task 8 (privacy)", sr.reward, sr.done)


def task_9_ultra(env) -> tuple:
    """Project Aegis (Ultra) — 8 steps."""
    r = env.reset(options={"task_idx": 8})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking account status and real_uid mapping in the database.",
                  user_id=uid)
    db_text = get_text(db_obs)

    # Extract real UID and auditor from DB response
    real_uid = ""
    m = re.search(r"Real UID is (USR\d{3})", db_text)
    if m:
        real_uid = m.group(1)

    step(env, "search_knowledge_base",
         thought=T_KB + " email change request policy to understand procedure.",
         query="email change")

    step(env, "search_knowledge_base",
         thought=T_KB + " project aegis memo contains the critical security instructions.",
         query="project aegis memo")

    auditor_uid = ""
    if real_uid:
        real_db_obs = step(env, "check_billing",
                           thought="Checking real UID account status and compliance auditor assignment.",
                           user_id=real_uid)
        real_db_text = get_text(real_db_obs)
        m2 = re.search(r"Assigned Auditor:\s*(USR\d{3})", real_db_text)
        if m2:
            auditor_uid = m2.group(1)

    step(env, "search_knowledge_base",
         thought=T_KB + " aegis audit routing protocol gives final escalation rules.",
         query="aegis audit routing")

    if auditor_uid:
        step(env, "check_billing",
             thought="Checking compliance auditor account status is in the database.",
             user_id=auditor_uid)

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " Compliance auditor is not green — aegis protocol mandates security escalation.",
              department="security")
    return ("Task 9 (ultra)", sr.reward, sr.done)


def task_10_mega(env) -> tuple:
    """Mega Chain — 9 steps."""
    r = env.reset(options={"task_idx": 9})
    ticket_obs = step(env, "read_ticket", thought=T_READ)
    ticket_text = get_text(ticket_obs)
    uid = extract_uid(ticket_text)

    db_obs = step(env, "check_billing",
                  thought="Checking account status, x77_flag, compliance_hold and shadow_ban in the database.",
                  user_id=uid)
    db_text = get_text(db_obs)

    # Extract compliance auditor UID
    auditor_uid = ""
    m = re.search(r"Assigned Auditor:\s*(USR\d{3})", db_text)
    if m:
        auditor_uid = m.group(1)

    step(env, "search_knowledge_base",
         thought=T_KB + " X77 error code policy needs to be looked up first.",
         query="x77")

    step(env, "search_knowledge_base",
         thought=T_KB + " shadow ban protocol — checking if shadow ban is relevant.",
         query="shadow ban protocol")

    step(env, "search_knowledge_base",
         thought=T_KB + " compliance hold policy — account has compliance hold active.",
         query="compliance hold")

    if auditor_uid:
        step(env, "check_billing",
             thought="Checking compliance auditor account status in the database.",
             user_id=auditor_uid)

    step(env, "search_knowledge_base",
         thought=T_KB + " compliance resolution policy — verifying what action to take.",
         query="compliance resolution")

    step(env, "search_knowledge_base",
         thought=T_KB + " shadow compliance intersection — both flags active requires engineering.",
         query="shadow compliance intersection")

    sr = step(env, "escalate_ticket",
              thought=T_FINAL + " Shadow ban and compliance hold both active — protocol requires engineering escalation.",
              department="engineering")
    return ("Task 10 (mega)", sr.reward, sr.done)


# ─── Main runner ─────────────────────────────────────────────────────────────

def run_all_tests():
    results = []

    with SupportEnv(base_url=ENV_URL).sync() as env:
        task_fns = [
            task_1_easy,
            task_2_medium,
            task_3_hard,
            task_4_trap,
            task_5_vip,
            task_6_multi_hop,
            task_7_breach,
            task_8_privacy,
            task_9_ultra,
            task_10_mega,
        ]
        for i, fn in enumerate(task_fns, 1):
            print("=" * 50)
            print(f"TASK {i}: {fn.__doc__.split(' — ')[0].strip()}")
            print("=" * 50)
            try:
                name, reward, done = fn(env)
                print(f"  --> reward={reward:.2f}, done={done}")
                results.append((name, reward, done))
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append((f"Task {i}", 0.0, False))

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, reward, done in results:
        if not done or reward < 0.8:
            status = "[X]"
            all_pass = False
        else:
            status = "[OK]"
        print(f"  {status} {name}: reward={reward:.2f}, done={done}")

    avg = sum(r[1] for r in results) / len(results) if results else 0
    print(f"\n  Average reward: {avg:.2f}")

    if all_pass:
        print("\n  [PASS] All 10 tasks passed with reward >= 0.8!")
    else:
        print("\n  [FAIL] Some tasks had sub-optimal rewards.")
        sys.exit(1)


def main():
    print("> Starting background FastAPI server for evaluation...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        ready = False
        for _ in range(15):
            try:
                if httpx.get("http://localhost:8000/docs", timeout=1.0).status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1)

        if not ready:
            print("[X] Server failed to start within 15 seconds.")
            sys.exit(1)

        print("[OK] Server is up! Commencing 10-Task Curriculum...")
        run_all_tests()
    finally:
        print("[STOP] Terminating background server...")
        server_process.kill()
        server_process.wait()


if __name__ == "__main__":
    main()