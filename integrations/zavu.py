"""Human approval gate over WhatsApp (Zavu).

Flow:
  1. The planner calls request_approval(): a `pending` row is inserted in `approvals`.
  2. A WhatsApp message asks the scientist to reply YES or NO.
  3. The Supabase Edge Function `zavu-webhook` receives the reply and updates the row.
     (The dashboard Approve/Deny buttons update the same row.)
  4. wait_for_decision() polls the row until it is approved/denied or times out.

In benchmark mode (campaigns with many seeds) approvals are auto-approved and logged.
"""
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from integrations.supabase_sync import db

load_dotenv()


def send_whatsapp(text, to=None):
    key = os.getenv("ZAVUDEV_API_KEY")
    to = to or os.getenv("APPROVER_PHONE")
    if not (key and to):
        print(f"[zavu offline] would send to {to}: {text}")
        return None
    r = requests.post(
        "https://api.zavu.dev/v1/messages",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"to": to, "text": text, "channel": "whatsapp"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def request_approval(run_id, round_, design, n_measurements, materials, reason, benchmark=False, notify=True):
    """notify=False: dashboard-only approval (judge rounds), no WhatsApp message."""
    request = (f"IonForge wants to spend {n_measurements} measurements on the *{design}* design "
               f"({', '.join(materials[:5])}) because {reason}")
    if benchmark:
        row = db.insert("approvals", {
            "run_id": run_id, "round": round_, "request": request, "design": design,
            "status": "auto_approved", "decided_by": "benchmark-policy", "channel": "benchmark",
            "decided_at": datetime.now(timezone.utc).isoformat(),
        })[0]
        return row
    row = db.insert("approvals", {"run_id": run_id, "round": round_, "request": request, "design": design})[0]
    approval_id = row.get("id", "offline")
    if notify:
        send_whatsapp(f"🔬 {request}\n\nReply *YES {approval_id}* or *NO {approval_id}*.")
    channel = "WhatsApp" if notify else "the dashboard"
    db.event(run_id, round_, "safety", "approval", f"Approval #{approval_id} requested via {channel}", {"approval_id": approval_id})
    return row


def wait_for_decision(approval_id, timeout_s=600, poll_s=3):
    """Returns 'approved', 'denied' or 'timeout'. Offline mode approves immediately."""
    if not db.enabled:
        return "approved"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rows = db.select("approvals", {"id": approval_id}, columns="status")
        if rows and rows[0]["status"] in ("approved", "denied", "auto_approved"):
            return "approved" if rows[0]["status"] != "denied" else "denied"
        time.sleep(poll_s)
    # Close the request so it does not stay pending forever (dashboard + judge runner read it).
    db.update("approvals", {"id": approval_id, "status": "pending"},
              {"status": "expired", "decided_at": datetime.now(timezone.utc).isoformat()})
    return "timeout"


if __name__ == "__main__":
    print(send_whatsapp("IonForge test message: the approval gate is wired up."))
