# Find the Promos — deterministic foundation

I built a source-preserving parser and a no-LLM candidate screener for Kuku Problem 1. It runs entirely offline with Python 3.10+ and the standard library. There are no packages to install, API keys, model calls or usage costs.

This is the requested audit/parser/candidate stage, not a finished semantic promo selector or the final two-page submission PDF. I used ChatGPT to help write and test the code. One concrete failure of the resulting rules is that “dead vegetation” gets treated as a hook. I preserve that failure in the raw output and explain it in the review.

## Run

Extract this bundle, open a terminal in `kuku_promos`, and run:

```bash
python run.py --input input --out outputs
python -m unittest discover -s tests -v
```

On Windows, use `py` instead of `python` if needed. From an existing Python notebook or Colab cell, prefix commands with `!`. No model/network access is needed after the files are available.

Run a **new screenplay pack** without overwriting the supplied run:

```bash
python run.py --input "path/to/new_scripts.zip" --out new_outputs
```

A directory of `.txt` files is also accepted. ZIP extraction rejects path traversal, symlinks, duplicate text member paths and oversized inputs. The tests are regression checks for the bundled 50 episodes; they are not assertions about a new show's episode count.

The stages can also run separately:

```bash
python parser.py --input input --out outputs --allow-errors
python candidates.py --episodes outputs/episodes.jsonl --rubric rubric.yaml --out outputs
python audit_supplement.py --input input --out outputs --name-prefix Ash
```

`parser.py` defaults to a nonzero exit when it finds errors, after saving its diagnostic outputs. In this pack there is one scene-containment error: Episode 22's SFX extends 0.30s into the following scene. `--allow-errors` preserves the source and permits inspection; it does not repair it. `run.py` permits this specific containment class but stops on other structural errors. `candidates.py` also rejects structurally broken input. The name-prefix supplement is an optional corpus-specific inspection aid; use a suitable prefix for another show.

## What I found

| Measure | Result |
|---|---:|
| Episodes | 50, numbered 1–50, no missing/duplicate numbers |
| Scenes / events | 100 / 3,727 |
| Source-clock runtime | 6,162.62s = 102m 42.62s |
| Min / median / max duration | 70.50 / 117.785 / 200.75s |
| Event types | 1,754 camera; 1,487 dialogue; 372 action; 114 SFX |
| Unparsed nonempty lines / missing event timestamps | 0 / 0 |
| Windows screened | 17,391 (12,828 single; 4,563 adjacent joins) |
| Eligible pool | 300 (219 single; 81 joins) |
| Selected after zero-overlap suppression | 51 (36 single; 15 joins) |

The parser records 161 notices: 88 format/order information entries, 72 warnings, and the single containment error. Episode 6 is the only episode with uppercase scene fields and section labels. There are 56 headers with extra speaker parentheses, 16 with voice cues in the final tone parentheses, 8 source-order time regressions, 20 same-track overlap pairs, and 52 uncovered timeline intervals. All occurrences are source-located in the audit. Some overlaps and gaps are intentional rather than errors.

I also found naming drift, including Ashbury/Ashburton within Episode 22. See `review/audit_notes.md`; names are never silently corrected.

## Files

| File | Purpose |
|---|---|
| `parser.py` | Parse source text, preserve text/time evidence, generate mechanical audit |
| `candidates.py` | Build temporal beats, enumerate windows/joins, score and suppress overlap |
| `rubric.yaml` | Definitions, weights, constraints, keyword rules and ranking config |
| `run.py` | One-command ZIP/directory pipeline |
| `audit_supplement.py` | Extra format/name-surface inventory |
| `reviewer.py`, `review_config.json` | Optional swappable review interface; default offline provider abstains |
| `tests/test_pipeline.py` | Real Episode 6 regression, corpus coverage and candidate invariant tests |
| `outputs/episodes.jsonl` | One complete parsed episode per line |
| `outputs/parse_report.md` | Audit summary, episode inventory and every mechanical exception |
| `outputs/audit_issues.jsonl`, `audit_summary.json` | Machine-readable audit and counts |
| `outputs/speaker_inventory.json`, `format_inventory.json` | Exact observed labels and formats |
| `outputs/name_surface_occurrences.jsonl` | Source-located name variants collected with the Ash prefix |
| `outputs/beats.jsonl` | Whole dialogue/action temporal components |
| `outputs/candidate_pool.jsonl` | All 300 screening survivors before overlap suppression |
| `outputs/candidates.jsonl` | Ranked 51 selected candidates with ordered segments and evidence |
| `outputs/candidate_stats.md`, `candidate_stats.json` | Generation/rejection counts and effective config |
| `outputs/*run.log`, `tests.log` | Raw command summaries and test output |
| `review/failure_review.md` | Eight inspected candidates, specific failure cases and a promising example |
| `input/` | Original extracted episode text, unchanged |
| `MANIFEST.sha256` | Checksums for the bundle's other files |

The raw `outputs/` files are generated by code; the editorial observations live separately in `review/`.

## Representation and exact timing

Each episode has `number`, `logline`, `continuity_notes`, `duration_sec` (also `duration`), scenes and ordered events. Every event contains the requested `episode`, `scene`, `type`, `speaker`, `tone`, `start_sec`, `end_sec`, `text`, plus a stable source ID, original header/time strings, line numbers, source order and delivery cues. The episode records the source filename and SHA-256.

Runtime is the maximum source scene/event end from zero, not the sum of overlapping tracks. Events sort by start time with original order breaking ties. The final parentheses are preserved as tone; earlier parentheses stay in speaker. A tone field containing only V.O./O.S. is a preserved notation variant, not an inferred emotion.

Beats are strictly overlapping dialogue/action components, not LLM-inferred story beats. Camera and SFX stay auxiliary so long sound beds do not merge a whole episode. A boundary cannot split dialogue or action. Each selected segment starts/ends on an actual event start/end, and its display timestamp is copied from that source event. Cuts may fall inside a camera shot; the score and review flag expose that. Timeline gaps are retained as real elapsed time.

Cross-episode candidates join the actual tail of Ep N to the actual opening of Ep N+1. They never skip missing episode numbers. Both pieces last 3–50s and their sum must be 20–60s.

## Ranking and hard-constraint limits

The five weighted components are hook 30, cliffhanger 30, standalone 20, emotional intensity 10 and clean cut 10. Each component is a documented heuristic in `evaluate()`. Scene-final/episode-final positions are recorded but cannot make a resolved scene into a cliffhanger.

I add `5 / first_episode_number` as a small landing bonus because I assume new viewers start at Ep 1. I subtract two points for each already-selected item touching the same episode. The greedy selection then suppresses all shared footage by default. Eight per episode is a ceiling, not a target that justifies padding. With 20s minimum clips and 70–201s episodes, eight disjoint single-episode clips are usually impossible.

For an exploration pool with near-duplicate suppression instead of zero shared footage:

```bash
python candidates.py --max-per-episode 8 --max-overlap 0.6 --out exploratory_outputs
```

This deliberately permits overlap and changes the guarantee. Overlap is shared source seconds divided by the shorter candidate's duration, including cross-episode segments. The default `candidates.jsonl` has **zero** shared footage between any pair.

I can enforce duration and source-derived, whole dialogue/action boundaries in code. I cannot prove hook impact, cold-viewer understanding or unresolved story state from keyword matches. The rule checks hook-event onset, not the exact word's delivery time. Questions and ellipses remain weak proxies. Every output is `needs_review`; the semantic hard constraints are required for final approval and are not represented as passed here.

`rubric.yaml` uses JSON syntax, which is valid YAML 1.2, to avoid a PyYAML dependency. Keep edits in JSON-compatible YAML syntax. The exact effective config is included in `candidate_stats.json`.

## Model interface and scale

The requested stage uses no LLM. `reviewer.py` is an optional extension point, not a provider integration claimed to have been tested. The default offline provider returns `not_reviewed`. To add a reviewer, implement `review(candidate, episodes) -> dict` in a local adapter and set `provider` to `your_module:YourClass` in `review_config.json`. Only load trusted modules. A future reviewer should select existing event IDs; source timestamps must continue to come from the parser. The candidate generator does not import or invoke an LLM.

For larger batches I would parse and screen each episode independently, keep one neighboring episode for joins, and send only a reduced pool to any semantic reviewer. The present implementation loads the supplied pack in memory. Its audit compares overlap pairs within episodes; window enumeration is quadratic in beats per episode, bounded by the 60s window; greedy diversity/overlap selection is quadratic in the retained pool. This is adequate for this pack but is not a claim of unlimited streaming scale. Local timing is in `outputs/run_manifest.json`; API cost for this run is zero.

## Before submission

I would validate the strongest clips with a cold viewer, replace proxy ranking with reviewed ranking, and then write the required two-page solution PDF. This bundle is not yet that final submission.

The assignment says the input pack is confidential. Keep this bundle private. If putting code on a public GitHub repository, exclude `input/`, `outputs/`, `review/` and the complete ZIP; they contain screenplay content. The included `.gitignore` guards those directories for a new repository but does not remove files that were already tracked.
