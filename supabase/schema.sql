-- IonForge Supabase schema (contract between the lab engine and the dashboard).
-- Run once in the Supabase SQL editor.
-- The lab writes with the service key; the dashboard reads with the anon key.

create table if not exists runs (
  id text primary key,
  strategy text not null,            -- ionforge | ablation_no_lit | ablation_anon | bo_prior | bo_cold | heuristic | random
  seed int,
  budget int,
  status text default 'running',     -- running | done | failed
  is_demo boolean default false,     -- seeded fake data; deleted before the demo
  created_at timestamptz default now()
);

create table if not exists events (
  id bigserial primary key,
  run_id text references runs(id) on delete cascade,
  round int,
  agent text,                        -- literature_scout_sulfides | ... | hypothesis_generator | screening | experiment_planner | safety | lab_runner | critic
  kind text,                         -- evidence | hypothesis | candidates | plan | approval | measurement | review | reopen | briefing
  summary text,
  payload jsonb,
  audio_url text,                    -- ElevenLabs round briefing (optional)
  created_at timestamptz default now()
);

create table if not exists hypotheses (
  id text primary key,
  run_id text references runs(id) on delete cascade,
  round int,
  statement text,
  predicted_effect text,
  confidence real,
  status text default 'open',        -- open | supported | rejected | reopened
  evidence_ids text[],
  parent_id text,
  created_at timestamptz default now()
);

create table if not exists measurements (
  id bigserial primary key,
  run_id text references runs(id) on delete cascade,
  round int,
  material text,
  family text,
  log_sigma real,
  superionic boolean,
  requested_by text,
  reason text,
  cumulative_found int,
  families_found int,
  n_measured int,
  created_at timestamptz default now()
);

-- Aggregated curves per strategy (median + IQR over seeds), written by lab/metrics.py.
create table if not exists curves (
  id bigserial primary key,
  strategy text,
  metric text,                       -- found | families
  n_measured int,
  median real,
  q25 real,
  q75 real,
  n_seeds int,
  is_demo boolean default false
);

create table if not exists evidence (
  id text primary key,
  doi text,
  title text,
  family text,
  trend text,
  quote text,
  verified boolean default false,
  blocked_leak boolean default false,
  blocked_reason text,
  source text,                       -- openalex | arxiv | brightdata
  is_demo boolean default false,
  created_at timestamptz default now()
);

create table if not exists approvals (
  id bigserial primary key,
  run_id text references runs(id) on delete cascade,
  round int,
  request text,
  design text,                       -- exploit | explore | test_hypothesis
  status text default 'pending',     -- pending | approved | denied | auto_approved
  decided_by text,
  channel text,                      -- whatsapp | dashboard | benchmark
  decided_at timestamptz,
  created_at timestamptz default now()
);

create table if not exists candidates (
  id text primary key,
  formula text,
  mp_id text,
  e_hull real,
  band_gap real,
  pred_log_sigma real,
  uncertainty real,
  in_domain boolean,
  domain_distance real,
  rationale text,
  rank int,
  next_experiment text,
  is_demo boolean default false
);

create index if not exists events_run_idx on events(run_id, round);
create index if not exists measurements_run_idx on measurements(run_id, n_measured);

-- Realtime
alter publication supabase_realtime add table events, measurements, hypotheses, approvals;

-- RLS: public read, writes only with the service key (which bypasses RLS).
do $$
declare t text;
begin
  foreach t in array array['runs','events','hypotheses','measurements','curves','evidence','approvals','candidates'] loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists "public read" on %I', t);
    execute format('create policy "public read" on %I for select to anon, authenticated using (true)', t);
  end loop;
end $$;

-- Dashboard Approve/Deny buttons: anon may only flip a pending approval.
drop policy if exists "dashboard decides approvals" on approvals;
create policy "dashboard decides approvals" on approvals for update to anon, authenticated
  using (status = 'pending')
  with check (status in ('approved', 'denied') and channel = 'dashboard');

-- Convenience view for the "X cards blocked by the no-leak rule" counter.
create or replace view evidence_stats as
select
  count(*) filter (where not blocked_leak) as passed,
  count(*) filter (where blocked_leak) as blocked,
  count(*) filter (where verified and not blocked_leak) as verified
from evidence;

-- Live campaign map (also in supabase/migrations/002_agent_activity.sql)
create table if not exists agent_activity (
  id bigserial primary key,
  run_id text not null,
  round int,
  agent text,
  state text,
  action text,
  tool text,
  detail jsonb,
  created_at timestamptz default now()
);
create index if not exists agent_activity_run_idx on agent_activity(run_id, created_at);
create or replace view agent_status as
select distinct on (run_id, agent) run_id, agent, round, state, action, tool, created_at
from agent_activity
order by run_id, agent, created_at desc;
alter table agent_activity enable row level security;
drop policy if exists "public read" on agent_activity;
create policy "public read" on agent_activity for select to anon, authenticated using (true);
alter publication supabase_realtime add table agent_activity;

-- Judge-triggered rounds: see supabase/migrations/003_judge_runs.sql (run_requests, run_requests_public, runner_status).

-- Spoken dashboard items: see supabase/migrations/004_audio_clips.sql (audio_clips).

-- Approvals: anon may only decide the running judge round; see supabase/migrations/005_judge_only_approvals.sql.
