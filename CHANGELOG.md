# Changes

- Stage A accepts no longer receive invented semantic flags, scores, or other required fields. Missing fields are logged by name. Only absent cliffhanger_why and accept reject_reason are repaired. Reject rows may be padded with empty fields. Model timestamps remain prohibited.
- Stage B requires all four fields, while retaining Yes/Y/true, extra-key and null-list tolerance. Its pass rule is unchanged: would keep watching and at most one missing-context item.
- Non-object provider responses enter the existing invalid-response retry path.
- Empty postprocessing writes both CSV headers and an explicit no-promos message.
- Colab defaults to screenplay ZIP upload; postprocessing now precedes export.
- README describes the current Stage B rule and source-only archive. Known limitations are retained.
- Original-corpus tests skip when absent or when only a partial/synthetic corpus is provided. Wide-pool tests use generated input counts rather than 750, handle non-contiguous episode numbers, and test keyword independence separately. Added regression tests for the four code fixes, logged missing flags, and timestamp rejection.

## Offline verification

- Clean source package, no input or outputs: 30 tests run, OK, 2 corpus/output-dependent class skips.
- Three synthetic screenplays and generated outputs: 35 tests run, OK, 1 original-corpus class skip.
- Mock end-to-end: 45 candidates, 2 ranked fixtures, no failed stages.
- Submitted live log: A_validated=63. Cached replay using supplied pre-fix source: 68. Cached replay after these fixes: 68. Missing semantic flags/scores caused no loss among these accepted clips.
- Both replays: 47 B survivors, 47 ranked, 114 cache hits, 5 B cache misses. Cache misses were blocked offline, not sent to Gemini. The five missing B responses correspond to recovered clips; the existing cached ranking covers the 47 existing survivors only.
- Replay used config.colab.json, outputs_colab/episodes.jsonl and candidate_pool.jsonl, and project-local .cache/llm with a dummy environment key. No network calls were permitted. Cache/raw transcripts and all screenplay inputs are excluded from this source ZIP.
- Parser source and config are unchanged. The model selection and editorial rules were not weakened to preserve the acceptance count.

## Submission audit
- Corrected README notebook cell count, input-upload instructions, configured-model description and stale next steps.
- Distinguished last-run cost from total attempts and labelled pricing unverified.
- Documented raw-output version and offline replay limits. Runtime code and raw results unchanged.
