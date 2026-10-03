"""Thin Supabase REST client used by the lab and the integrations.

Writes use the service key (bypasses RLS). If SUPABASE_URL is not set, every
call becomes a no-op that prints the row, so the lab runs fully offline.

    from integrations.supabase_sync import db
    db.insert("events", {"run_id": "r1", "round": 1, "agent": "critic", "kind": "review", "summary": "..."})
"""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()


class Supabase:
    def __init__(self, url=None, key=None):
        self.url = (url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.key = key or os.getenv("SUPABASE_SERVICE_KEY")
        self.enabled = bool(self.url and self.key)

    def _headers(self, prefer="return=representation"):
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        }

    @staticmethod
    def _columns(rows):
        """Union of keys, so bulk rows with different fields share one column list.
        With Prefer: missing=default, absent keys take the column default."""
        rows = rows if isinstance(rows, list) else [rows]
        return ",".join(dict.fromkeys(k for r in rows for k in r))

    def _req(self, method, table, **kw):
        r = requests.request(method, f"{self.url}/rest/v1/{table}", timeout=30, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase {method} {table} failed: {r.status_code} {r.text}")
        return r.json() if r.text else []

    def insert(self, table, rows):
        if not self.enabled:
            print(f"[supabase offline] {table}: {json.dumps(rows, default=str)[:300]}")
            return rows if isinstance(rows, list) else [rows]
        return self._req(
            "POST", table,
            headers=self._headers("return=representation,missing=default"),
            params={"columns": self._columns(rows)},
            data=json.dumps(rows, default=str),
        )

    def upsert(self, table, rows, on_conflict="id"):
        if not self.enabled:
            return self.insert(table, rows)
        return self._req(
            "POST", table,
            headers=self._headers("resolution=merge-duplicates,return=representation,missing=default"),
            params={"on_conflict": on_conflict, "columns": self._columns(rows)},
            data=json.dumps(rows, default=str),
        )

    def update(self, table, match, values):
        """match: dict of column -> value (equality filters)."""
        if not self.enabled:
            print(f"[supabase offline] update {table} {match}: {values}")
            return []
        params = {k: f"eq.{v}" for k, v in match.items()}
        return self._req("PATCH", table, headers=self._headers(), params=params, data=json.dumps(values, default=str))

    def select(self, table, match=None, columns="*", order=None, limit=None):
        if not self.enabled:
            return []
        params = {"select": columns, **{k: f"eq.{v}" for k, v in (match or {}).items()}}
        if order:
            params["order"] = order
        if limit:
            params["limit"] = limit
        return self._req("GET", table, headers=self._headers(), params=params)

    def delete(self, table, filters):
        """filters: raw PostgREST filters, e.g. {"is_demo": "eq.true"}."""
        if not self.enabled:
            return []
        return self._req("DELETE", table, headers=self._headers(), params=filters)

    def event(self, run_id, round_, agent, kind, summary, payload=None, audio_url=None):
        return self.insert("events", {
            "run_id": run_id, "round": round_, "agent": agent, "kind": kind,
            "summary": summary, "payload": payload or {}, "audio_url": audio_url,
        })


db = Supabase()
