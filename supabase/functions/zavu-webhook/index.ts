// Supabase Edge Function: receives Zavu `message.inbound` webhooks and decides approvals.
// Reply format on WhatsApp: "YES 12" / "NO 12" (or just "YES"/"NO" for the latest pending one).
//
// Deploy:  supabase functions deploy zavu-webhook --no-verify-jwt
// Secrets: supabase secrets set ZAVU_WEBHOOK_SECRET=... APPROVER_PHONE=+569...
// Then set the webhook URL in the Zavu dashboard (sender -> webhook, event message.inbound):
//   https://<project-ref>.supabase.co/functions/v1/zavu-webhook
import { createClient } from "npm:@supabase/supabase-js@2";

const SECRET = Deno.env.get("ZAVU_WEBHOOK_SECRET") ?? "";
const APPROVER = Deno.env.get("APPROVER_PHONE") ?? "";
const supabase = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);

async function hmacHex(key: string, msg: string): Promise<string> {
  const k = await crypto.subtle.importKey("raw", new TextEncoder().encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, new TextEncoder().encode(msg));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function verify(rawBody: string, header: string | null): Promise<boolean> {
  if (!SECRET) return true; // signature check disabled until the secret is set
  if (!header) return false;
  const parts: Record<string, string> = {};
  for (const piece of header.split(",")) {
    const i = piece.indexOf("=");
    if (i > 0) parts[piece.slice(0, i).trim()] = piece.slice(i + 1).trim();
  }
  const age = Math.floor(Date.now() / 1000) - Number(parts.t);
  if (!(age <= 300 && age >= -60)) return false;
  const expected = parts.v2 ? await hmacHex(SECRET, `${parts.t}.${rawBody}`) : await hmacHex(SECRET, rawBody);
  const received = parts.v2 ?? parts.v1 ?? "";
  return safeEqual(expected, received);
}

Deno.serve(async (req) => {
  const raw = await req.text();
  if (!(await verify(raw, req.headers.get("X-Zavu-Signature")))) {
    return new Response("invalid signature", { status: 401 });
  }
  const event = JSON.parse(raw);
  if (event.type !== "message.inbound") return new Response("ignored");

  const { from, text } = event.data ?? {};
  if (APPROVER && from !== APPROVER) return new Response("unknown sender");

  const m = String(text ?? "").trim().toUpperCase().match(/^(YES|SI|SÍ|NO)\b\s*#?(\d+)?/);
  if (!m) return new Response("no decision");
  const status = m[1] === "NO" ? "denied" : "approved";

  let id = m[2] ? Number(m[2]) : null;
  if (id === null) {
    const { data } = await supabase.from("approvals").select("id").eq("status", "pending")
      .order("created_at", { ascending: false }).limit(1);
    id = data?.[0]?.id ?? null;
  }
  if (id === null) return new Response("nothing pending");

  const { error } = await supabase.from("approvals")
    .update({ status, channel: "whatsapp", decided_by: from, decided_at: new Date().toISOString() })
    .eq("id", id).eq("status", "pending");
  if (error) return new Response(error.message, { status: 500 });
  return new Response(`approval ${id} ${status}`);
});
