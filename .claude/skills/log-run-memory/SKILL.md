---
name: log-run-memory
description: Write the standardized post-run eval/training memory entry plus its one-line MEMORY.md index pointer. Use after an eval or training run has been interpreted and the result should be persisted.
---

# log-run-memory

## When to trigger

- Immediately after `nav-eval` produces a verdict ("save this", "remember this", or implicitly as the eval handoff)
- After `ppo-triage` identifies a non-obvious root cause worth persisting
- "log this run" / "update memory with the result"
- A run finishes and the user moves on without saying — offer to persist if the result changes future decisions

Do NOT trigger to record what git/code/CLAUDE.md already states (code structure, the launch command itself, a routine pass). Persist the non-obvious *why*, not the event.

## File

`memory\nav_phaseX_vN_eval.md` (or a topic slug for a lesson). One fact per file. Frontmatter:

```markdown
---
name: <short-kebab-or-phrase>
description: <one-line summary used for recall>
metadata:
  type: project        # feedback / reference / user as appropriate
  originSessionId: <current session id if known>
---
```

## Body template (eval result)

- Header: `**Phase X vN eval (M/D/2026)**: <N>M steps from <source>, <what changed>.`
- Table: `| Maze | prev | this | Δ | Pass |` for corridor/L/U/procedural + composite row.
- `**Why:**` one line — the mechanism.
- `**How to apply:**` concrete next action(s).
- `**Verdict:**` PASS / FAIL / HOLLOW + the deciding number.
- `[[slug]]`-link related memories (prior version, root-cause notes) liberally.

## Rules

- Absolute dates (today resolvable from context); no "yesterday".
- Existing file covering this run/lesson → UPDATE, never duplicate. Delete memories proven wrong.
- One fact per file.

## Index pointer (required)

Append ONE line to `memory\MEMORY.md` under the right section:
`- \`nav_phaseX_vN_eval.md\` — **<terse hook>**`

## Verification step (before finishing)

1. **Frontmatter parses.** Re-read the written file: `---` fences balanced, `name`/`description`/`type` present. A malformed header makes the memory unrecallable.
2. **No duplicate.** Glob `memory\nav_phase*` (or grep the topic) BEFORE writing; if a file covers this run, you must be editing it, not creating `_v2`. Confirm post-write that only one file owns this fact.
3. **MEMORY.md line budget.** MEMORY.md is **over its 200-line limit**. Count lines after the edit; if you added a line, you MUST prune/merge a stale superseded line (e.g. an old vN now superseded) in the SAME edit so net count does not grow. Verify the count did not increase.
4. **Index ↔ file consistency.** The new MEMORY.md pointer's filename must exactly match the file you wrote, and its hook must match the verdict. Mismatched index entries are worse than none.
5. **Links resolve or are intentional.** Each `[[slug]]` either matches an existing file or is a deliberate forward-marker — no accidental typos of real slugs.

Only after 1–5: report what was saved (file + one-line index entry) tersely.

## Tools this skill needs

- **Write / Edit** — create the memory file; **Edit** (not Write) on MEMORY.md to append/prune without rewriting the whole index.
- **Glob / Grep** — verification 2 (dedupe scan) and 5 (link resolution) before writing — the single most important check; duplicate/contradictory memories are the documented failure mode (cf. the retracted contamination note that caused 3 failed runs).
- **Read / Bash line-count** — verification 3 (MEMORY.md 200-line budget).
- No external/MCP tool needed; this skill is fully local and deterministic.
