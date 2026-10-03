-- Judge-triggered rounds: a public "Run one round" button queues a request; the judge
-- runner (scripts/judge_runner.py, on a laptop or in GitHub Actions) claims it atomically,
-- runs one round of the shared live campaign, and reports back.

create table if not exists run_requests (
  id bigserial primary key,
  status text not null default 'queued'
    check (status in ('queued', 'running', 'done', 'expired', 'denied', 'failed', 'cancelled', 'rejected')),
  source text default 'dashboard',
  requested_at timestamptz not null default now(),
  claimed_at timestamptz,
  finished_at timestamptz,
  executor text,          -- who ran it: 'laptop-<host>' | 'github-actions'
  run_id text,            -- the live campaign the round belongs to
  round int,
  note text               -- short outcome for the dashboard, e.g. "Measured 5: 1 hit"
);
create index if not exists run_requests_status_idx on run_requests(status, requested_at);

-- Anyone with the dashboard URL may queue a request, and nothing else:
-- anon can insert only the `source` column (defaults fill the rest), cannot read the
-- table, cannot update or delete. The runner uses the service key.
alter table run_requests enable row level security;
revoke all on run_requests from anon, authenticated;
grant insert (source) on run_requests to anon, authenticated;
grant usage on sequence run_requests_id_seq to anon, authenticated;
drop policy if exists "queue a round" on run_requests;
-- No column checks needed (they would require SELECT): anon can write only `source`,
-- so status/executor/run_id always take their defaults ('queued', null, null).
create policy "queue a round" on run_requests for insert to anon, authenticated with check (true);

-- Anti-spam: at most 10 requests waiting at once.
-- security definer: anon cannot read run_requests, so the count must run as the owner.
create or replace function run_requests_cap() returns trigger language plpgsql
  security definer set search_path = public as $$
begin
  if (select count(*) from run_requests where status = 'queued') >= 10 then
    raise exception 'The queue is full; try again in a few minutes.';
  end if;
  return new;
end $$;
drop trigger if exists run_requests_cap on run_requests;
create trigger run_requests_cap before insert on run_requests for each row execute function run_requests_cap();

-- What the dashboard may read (RLS filters rows, not columns, so a view exposes the safe ones).
create or replace view run_requests_public as
select id, status, requested_at, claimed_at, finished_at, round, run_id, note,
       case when status = 'queued'
            then (select count(*) from run_requests q where q.status = 'queued' and q.id <= r.id)
       end as queue_position
from run_requests r
order by id desc;
grant select on run_requests_public to anon, authenticated;

-- Runner heartbeat: the dashboard shows "Lab offline, watch the replay" when it goes stale.
create table if not exists runner_status (
  id text primary key,            -- 'judge-runner'
  executor text,
  enabled boolean default false,  -- JUDGE_RUNS switch as seen by the runner
  heartbeat_at timestamptz,
  current_run_id text,
  rounds_last_hour int default 0,
  rounds_today int default 0,
  max_per_hour int,
  max_per_day int,
  message text
);
alter table runner_status enable row level security;
drop policy if exists "public read" on runner_status;
create policy "public read" on runner_status for select to anon, authenticated using (true);

alter publication supabase_realtime add table runner_status;
