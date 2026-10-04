-- Security: the public dashboard may only decide approvals of the judge round that is
-- running right now (shared judge campaign, created < 3 minutes ago, still pending).
-- Live-demo and Materials Project approvals are decided by the scientist on WhatsApp
-- (Edge Function with the service key), never by an anonymous visitor.
drop policy if exists "dashboard decides approvals" on approvals;
create policy "judges decide their own round" on approvals for update to anon, authenticated
  using (
    status = 'pending'
    and created_at > now() - interval '3 minutes'
    and run_id = (select current_run_id from runner_status where id = 'judge-runner')
  )
  with check (status in ('approved', 'denied') and channel = 'dashboard');

-- Lovable had added an unrestricted policy; removed so only the judge-round rule remains.
drop policy if exists "dashboard approves" on approvals;
