# Lovable prompt

Connect the Supabase project in Lovable first (Integrations → Supabase), run `supabase/schema.sql`, then paste:

```
Build "IonForge", a live dashboard for an agentic AI lab that discovers solid-state lithium battery electrolytes. Read from the connected Supabase tables (runs, events, hypotheses, measurements, curves, evidence, evidence_stats view, approvals, candidates) with realtime subscriptions on events, measurements, hypotheses and approvals. All UI text in English.

Sections:
1. Header: research question "Can an agentic lab find superionic Li conductors with fewer lab measurements?", budget used (max n_measured / runs.budget for the selected run), current round, run selector (default: latest run with strategy = 'ionforge').
2. Live Lab: vertical timeline of events for the selected run, grouped by round; agent name as a colored chip (Literature Scout ×3, Hypothesis Generator, Screening, Experiment Planner, Safety, Lab Runner, Critic, Lab Voice), summary, expandable JSON payload. Show the 3 literature scout events side by side. For experiment_planner events, render payload.designs (exploit / explore / test_hypothesis) as 3 small score bars and highlight payload.chosen. Highlight kind = 'reopen' in red. If audio_url is set, show a ▶ play button.
3. Hypotheses: cards with statement, predicted effect, confidence bar, status pill (open / supported / rejected / reopened), evidence count, and a parent→child tree using parent_id.
4. Discovery curve: from the curves table, x = n_measured, y = median; one line per strategy (ionforge, bo_prior, bo_cold, heuristic, random) with a shaded band between q25 and q75. Toggle metric between "found" (superionic found) and "families" (distinct families found). Big number card: "N× fewer measurements than BO + prior knowledge to find k" (read from a row in runs or compute from curves: first n_measured where median >= k). Also overlay the selected run's own measurements.cumulative_found as dots.
5. Evidence: table with DOI link (https://doi.org/{doi}), family, trend, verbatim quote, verified badge. Hide rows with blocked_leak = true from the table, but show a counter from evidence_stats: "X cards blocked by the no-leak rule" with a tooltip that lists blocked_reason.
6. Approvals: list pending / approved / denied / auto_approved with channel (whatsapp / dashboard / benchmark) and decided_at. Pending rows have Approve and Deny buttons that update the row: status = 'approved' or 'denied', channel = 'dashboard', decided_at = now().
7. Next experiment: two tables from candidates, "In domain" (in_domain = true) and "Out of domain", with formula, Materials Project id (link to https://next-gen.materialsproject.org/materials/{mp_id}), E_hull, band gap, predicted log σ ± uncertainty drawn as wide error bars, domain distance, rationale, next_experiment. Banner: "AI-generated hypotheses, not lab-validated".

Style: clean scientific instrument look, light and dark mode, monospace numbers, crimson (#C4123A) for discoveries, teal (#0D6A88) for agents. Mobile friendly. Empty states explaining what will appear in each section.
```

Before the demo: `python scripts/clear_demo_data.py`.
