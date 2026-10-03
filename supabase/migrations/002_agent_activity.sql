-- Live campaign map: one row per agent action (tool call) written by lab/mcp_server.py.
-- Lets the dashboard show who is working right now and replay a finished campaign.
create table if not exists agent_activity (
  id bigserial primary key,
  run_id text not null,
  round int,
  agent text,                        -- pi | literature_scout | hypothesis_generator | screening | experiment_planner | safety_officer | critic
  state text,                        -- working | done | waiting
  action text,                       -- human-readable, e.g. "Ranking candidates with the uncertainty-aware surrogate"
  tool text,                         -- MCP tool name
  detail jsonb,
  created_at timestamptz default now()
);
create index if not exists agent_activity_run_idx on agent_activity(run_id, created_at);

-- Latest action per agent and run (drives the "who is working now" map).
create or replace view agent_status as
select distinct on (run_id, agent) run_id, agent, round, state, action, tool, created_at
from agent_activity
order by run_id, agent, created_at desc;

alter table agent_activity enable row level security;
drop policy if exists "public read" on agent_activity;
create policy "public read" on agent_activity for select to anon, authenticated using (true);

alter publication supabase_realtime add table agent_activity;
