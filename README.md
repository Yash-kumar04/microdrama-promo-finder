# Find the Promos: finds 20-60s promo clips in microdrama screenplays

Pipeline: **parser** (code) -> **wide candidate generator** (code, up to 15 candidates per episode, including episode endings and cross-episode joins) -> **Stage A model review** (picks and scores) -> **code validator** (every timestamp is derived from parsed event IDs, never typed by the model) -> **Stage B cold-viewer test** -> **Stage C ranking** -> **overlap removal** (`postprocess_diverse.py`) -> human editor.

## Real run on the 50-episode pack (Gemini `gemini-3.5-flash-lite`, free tier)
750 candidates -> 63 passed validation -> 47 passed the cold-viewer test and were ranked -> 35 after removing 12 overlapping clips. 0 failed stages. The last run logged about $0.009 of new paid-equivalent cost; this excludes earlier attempts and cached work. Rates in `config.yaml` are unverified and reference a different model. All clips are 20-60s.

## Known limitations (found by reading the outputs)
- Judged from screenplay text only, never video. A human editor must approve every promo.
- Hook/cliffhanger labels are often camera or sound text instead of the story beat (12 of 47 hooks and 22 of 47 cliffhanger labels are not spoken lines).
- Screenplays end with "cut to black / cliffhanger" cues, so 24 of 47 picks end on the episode's last second. Some of these are relief or atmosphere, not a real unresolved question (example: Ep 5 00:54-01:50.5).
- The first live run used a Stage B rule so strict that 0 of 48 clips could pass. I loosened it (viewer would keep watching AND at most one missing-context blocker) and kept the first run in `outputs_llm_colab_run1_strict_gate/`.
- **Output version note:** the raw outputs in the submission were produced BEFORE patch 2 (tolerant Stage A row keys). In that run 116 of 750 model decisions (8 episodes mostly affected) were thrown away because the model omitted a field such as `cliffhanger_why`; 7 of those were model-accepted clips. An offline replay of the cached model replies with the patch recovers 5 of them (Ep 44, 45, 48); re-running Stage B/C on them needs about 6 API calls. This repo contains the patched code.
- **3-second hook rule is strict:** a hook line must finish inside the first 3 seconds. In Ep 25 the model accepted two clips and the code rejected both for this rule, although a long opening action line is a normal hook in these scripts. A looser rule (hook starts within 3s, ends within 5s) is the next thing to test.
- **Answer-key check (LLM-suggested, 5 episodes, 15 candidates; all timestamps verified against the scripts):** the tool overlapped the key (IoU >= 0.5) in 0 of 5 best-of-3 picks and 1 of 5 when any of the 3 candidates counts (Ep 12: IoU 0.44-0.64). The tool found no promo at all in Ep 25 and Ep 38. The key prefers opening hooks and the tool prefers episode endings, so disagreement is not always a tool error, but Ep 25 is.
- Gemini model names change over time. If you see HTTP 404, set a currently available model in `config.yaml`.
- The screenplays are Kuku content and are NOT in this repo. Put them in `input/`.

## Run offline in one command

Python 3.10+ is sufficient. No packages or keys are required for the default JSON-compatible YAML config.

```bash
python run.py --input input --out outputs --llm --provider mock --llm-out outputs_llm
python -m unittest discover -s tests -v
```

On Windows I can use `py` in place of `python`. For a new screenplay ZIP:

```bash
python run.py --input "new_scripts.zip" --out new_outputs --llm --provider mock --llm-out new_outputs_llm
```

The input must be a screenplay-text ZIP or directory, not the full project ZIP (which also contains non-screenplay `.txt` files). Extract the project first, then use its `input/` directory. All source times/text remain grounded in the parsed input.

## Switch models in 60 seconds

`config.yaml` is JSON syntax, valid YAML 1.2. It works without dependencies. For ordinary indented YAML, install `PyYAML>=6,<7` first. Keep keys out of this file.

**Gemini:** the supplied config already has the Gemini model, base URL and environment-variable name. Set `provider` to `gemini`, or override it on the command line. Set `GEMINI_API_KEY` in the environment without saving it to a file. For a local terminal, this hidden prompt avoids shell history:

```bash
python -c "import os,getpass,subprocess,sys; os.environ['GEMINI_API_KEY']=getpass.getpass('Gemini API key: '); subprocess.run([sys.executable,'llm_layer.py','--provider','gemini','--out','outputs_llm_gemini'],check=True)"
```

To choose a different Gemini model, change only `model`, then update the pricing fields. The supplied model is `gemini-3.5-flash-lite`; availability and quota depend on the account. The pricing reference is a different model and must be checked before budgeting.

**Any endpoint implementing OpenAI Chat Completions:** edit these four config values:

```json
{
  "provider": "openai_compatible",
  "model": "YOUR_ENDPOINT_MODEL_ID",
  "base_url": "http://localhost:1234/v1",
  "api_key_env": ""
}
```

Merge these values into the existing config; do not replace the complete file with this fragment. For a hosted endpoint, use its HTTPS `/v1` base URL and the name of its key environment variable. An empty `api_key_env` allows an unauthenticated local server. Set `openai_json_mode` false if the endpoint lacks JSON mode; the prompt and local schema validation still require JSON. Change `openai_token_parameter` to `max_completion_tokens` if the chosen endpoint requires it. Provider-specific request settings can go in `provider_options` (Gemini generationConfig fields, or OpenAI body fields). Reset incompatible options when switching providers.

```bash
python llm_layer.py --config config.yaml --out outputs_llm_real
```

There is **no automatic fallback to mock**, no provider switching on quota errors, and no paid-key provisioning. Gemini free-tier usage is free only where eligible and within quota. The code cannot tell whether a key's project has billing enabled. A local OpenAI-compatible model is another free option, subject to local hardware and model quality.

## What changed in candidate generation

- Keywords only add points; a missing keyword never rejects a valid-duration window. Question marks, ellipses and resolution words are also not hard gates at this recall stage.
- I enumerate sliding 20–60s windows and explicit **episode endings: the last 20–60s**, all at parsed event boundaries.
- Adjacent joins are **last 15–40s of Ep N + first 5–20s of Ep N+1**, with a 20–60s total.
- The final camera/action event gets a **25-point bonus** when it contains cliffhanger, cut to black, abrupt cut, holds or fade wording. This is a selection prior, never semantic proof.
- The pool reserves up to four endings and four joins, then fills to 15 with a soft overlap penalty. Overlap is intentional: a judge needs alternate cuts. No valid option is invented if a new episode is too short or lacks safe boundaries.
- `outputs/candidate_pool.jsonl` and `outputs/candidates.jsonl` both contain the selected wide pool. Counts for all enumerated windows are in `candidate_stats.json`.

## Three-stage review

**A — episode editor.** One call per episode receives its logline, continuity notes, adjacent context if needed, a deduplicated event-text dictionary and the 15 candidate definitions. Source timestamps are not sent. Every candidate gets an acceptance or rejection; at most two are accepted by default. The model chooses candidate IDs, optionally refining cuts using existing `start_event_id`/`end_event_id` pairs. It returns exact hook/cliffhanger excerpts and reasons, one sentence on why the clip works, four 0–5 scores, resolved-goal/rhetorical-question flags, and rejection reasons. The prompt explicitly rejects resolved goals and rhetorical questions; code rejects any positive flags and hook/cliffhanger/standalone scores below 3.

**B — cold viewer.** Each validated acceptance gets a separate, stateless call containing **only clip text**. No logline, series context, event IDs, Stage A reasoning or scores are included. It answers what is confusing, missing context, whether it would keep watching, and what might happen next. All four reply fields must be present. A clip passes when the viewer would keep watching AND has at most one missing-context item. Confusing findings are retained for review, not used as an extra pass gate. Formatting differences, extra keys, and null lists are tolerated. A model cold-viewer simulation is not a real audience study.

**C — ranker.** One call receives all B survivors, their evidence/scores, source event IDs for overlap awareness, and a series summary formed from all loglines. It must return an exact permutation with reasons plus the first three to post. The weights are 30/30/20/10/10, with clean cut computed by code. Prompts request diversity and a modest early landing bonus, explicitly assuming viewers start at Ep 1. Bad/incomplete rankings are rejected; they are not silently replaced with a heuristic ranking. Unranked survivors are saved separately.

## Validation and honest limits

I validate IDs against the actual episode/event maps, not model text. I derive segment timestamps again from boundary event IDs. I reject unknown IDs, out-of-context boundaries, reversed/overlapping segments, nonadjacent episode joins, durations outside 20–60s, cuts inside dialogue/action, invented quotes, invalid scores, and cliffhanger evidence far from the end or followed by another semantic event.

**First-3s rule:** an entire cited hook event must finish within the opening three seconds. This conservatively proves that its quoted portion is inside the limit; event onset alone cannot prove when a later phrase is spoken. This misses valid hooks embedded in long lines. I will not invent word timestamps to improve recall.

Only complete events are included in clip text. A camera or SFX annotation crossing a cut could describe action outside the clip, so it is excluded from model evidence. Original complete/partial events remain traceable in segment event IDs. Cuts inside camera shots remain possible and lower the computed clean-cut score.

Quoted evidence must be an exact nonempty substring of its cited, wholly in-clip event. Names and stage directions can reveal context that video viewers would not know; human footage review is still needed. Semantic quality is model-attested, never proven by JSON validity. The baseline failure examples in `review/` are preserved for comparison; no claim is made that mock mode fixes them.

## Outputs and reproducibility

| Path | Purpose |
|---|---|
| `parser.py` | Unchanged parser, source evidence and audit |
| `outputs_baseline/` | Original supplied outputs, unchanged |
| `candidates_baseline.py`, `README_baseline.md` | Original strict implementation and documentation |
| `tests_original.py` | Original test source, unchanged archival copy |
| `tests/test_pipeline.py` | Original assertions; only candidate import redirected to baseline module |
| `tests/test_upgrade.py` | New pool, validation, context isolation, adapter, retry and cache tests |
| `outputs/` | Reproducible parsed episodes, 750-option pool, audit and stats |
| `llm_layer.py`, `llm_providers.py` | A/B/C orchestration, local validators, adapters, cache and logging |
| `config.yaml` | Provider/model/endpoint/key-variable, retry, budget and pricing settings |
| `outputs_llm/promos_ranked.json` | Metadata wrapper, top_3 IDs and ranked promos with grounded segments |
| `outputs_llm/promos_ranked.csv` | Requested flat columns, plus explicit mode/status |
| `outputs_llm/promos_report.md` | Summary, top three, failures and limits |
| `outputs_llm/rejected_candidates.csv` | Model, local validation, cold-viewer and API-stage rejection reasons |
| `outputs_llm/run_log.json` | Calls, attempts, cache hits, usage/cost estimates and failures |
| `outputs_llm/raw/` | Unedited prompts and provider-response attempts, without API keys |
| `outputs_llm/survivors_unranked.json` | B survivors even if C fails |
| `.cache/llm/` | Content-addressed successful JSON responses for replay (created when run) |
| `run_colab.ipynb` | Six code cells: setup/upload, secret/config, input upload, run, postprocess, export |
| `estimate_cost.py`, `cost_estimate.json` | Re-runnable 50/500-episode planning calculation |

`segments` in the CSV looks like `EP12 00:41.20-01:03.00 | EP13 00:00.00-00:14.50`. These are format examples, not selected source clips. Actual strings copy source precision, including hundredths; I do not round timings to tenths. JSON keeps exact numeric source seconds as well.

Offline generation and mock content are deterministic. Live requests use temperature zero but providers need not be deterministic. Cache keys include provider, model, endpoint, request settings, prompt/version and payload; changed input/model/prompt gets a different key. Replays re-run local validation. Cache-hit/token-usage fields in `run_log.json` intentionally differ between first and replayed runs; ranked JSON/CSV and rejection content should match.

Malformed JSON, transient network failures, HTTP 429 and selected 5xx errors get bounded retries. `Retry-After` is respected. If it exceeds the sleep budget, the live client stops sending requests instead of hammering a daily quota. Auth failures also stop further live requests. Re-running resumes through cached successes. Logs separate cache hits from newly estimated cost and include tokens spent on malformed/truncated responses. If usage is absent, counts are labelled character-based estimates. Failed network requests with no usage response may have unknown server-side charges.

To replay the old strict generator into a separate folder:

```bash
python candidates_baseline.py --episodes outputs_baseline/episodes.jsonl --out baseline_recheck
```

Do not point new outputs at `outputs_baseline/`.

## Cost estimate

I used actual Stage A prompt sizes from the 50-episode pack, the largest clip transcript per episode for Stage B, and an allowance for the global ranker. The default cap is two Stage A acceptances per episode. Character-to-token estimates include 25% input margin; output allowance is 6,000 tokens/episode for A, 400/clip for B and 100/survivor for C. There is no claim these are measured live tokens.

Using assumed rates of **$0.10/M input and $0.40/M output** (not verified for the configured model):

| Episodes | Maximum normal calls | Input estimate | Output allowance | Paid equivalent | With 25% retry allowance |
|---:|---:|---:|---:|---:|---:|
| 50 | 151 | 556,689 | 350,000 | $0.20 | $0.24 |
| 500 | 1,501 | 5,561,260 | 3,500,000 | $1.96 | $2.45 |

**Mock costs $0.** Eligible free-tier live calls cost $0 within quota, but the quota may prevent finishing a full run in one day. Model changes, thinking tokens and retries can change costs. Update `pricing` when changing models. `run_log.json` is a cost estimate, not an invoice; actual live charges are unknown to the tool.

The 500-episode numbers extrapolate this show's density. Stage C still receives all survivors, so a 500-episode run may need higher context/output limits or a larger-capacity model; budget violations/truncation are logged rather than dropping promos. No 500-episode live quality/throughput claim is made.

```bash
python estimate_cost.py
```

Primary API references: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [Gemini generateContent](https://ai.google.dev/api/generate-content), [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output), [OpenAI Chat Completions reference](https://platform.openai.com/docs/api-reference/chat/create). The adapters use HTTP directly and need no provider SDK.

## Colab and submission

Open `run_colab.ipynb` in Colab and run the cells. Upload this project ZIP first. For Gemini, create a Colab secret named `GEMINI_API_KEY` and enable notebook access; the notebook reads it through `google.colab.userdata` without printing it. The default notebook mode is Gemini; explicitly change `MODE` to `mock` for an offline run. It does not silently fall back if the secret is missing. Cell 3 requires your separate screenplay ZIP by default; no screenplay pack is bundled.

This source archive contains no screenplay input or API keys. Generated outputs and caches can contain screenplay text; keep those private. For a public code repository exclude `input/`, output directories, raw prompts, cache, review excerpts and ZIPs. The included `.gitignore` helps with new files but cannot untrack existing commits.

The full live run and first failed run are supplied separately as private raw evidence. They predate the current validation fixes. The final audit replay recovers 68 Stage A acceptances; five recovered clips still need live Stage B calls. No new live quality result is claimed. The source archive deliberately excludes generated outputs, caches and screenplay excerpts; paths in the outputs table describe generated or historical files, not bundled evidence.
