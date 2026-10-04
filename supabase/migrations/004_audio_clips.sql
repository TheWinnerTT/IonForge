-- Spoken versions of dashboard items (ElevenLabs), generated once by scripts/voice_worker.py
-- and cached in the public `briefings` bucket. The dashboard only plays them: judges can
-- never trigger a paid call.
create table if not exists audio_clips (
  id bigserial primary key,
  kind text not null,          -- evidence | hypothesis | event (review / reopen / next_step / plan)
  ref_id text not null,        -- evidence.id, hypotheses.id or events.id
  text_hash text not null,     -- regenerate only if the text changes
  url text not null,
  chars int,
  created_at timestamptz default now(),
  unique (kind, ref_id)
);
alter table audio_clips enable row level security;
drop policy if exists "public read" on audio_clips;
create policy "public read" on audio_clips for select to anon, authenticated using (true);
alter publication supabase_realtime add table audio_clips;
