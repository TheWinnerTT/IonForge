"""Remove every row created by seed_demo_data.py. Run before the real demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.supabase_sync import db  # noqa: E402

if not db.enabled:
    sys.exit("Supabase not configured")
# Deleting demo runs cascades to events, hypotheses, measurements and approvals.
for table in ("runs", "curves", "evidence", "candidates"):
    db.delete(table, {"is_demo": "eq.true"})
print("Demo data removed.")
