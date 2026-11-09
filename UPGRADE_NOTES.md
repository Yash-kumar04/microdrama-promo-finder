# Upgrade verification

I preserved parser.py and all 18 original output files byte-for-byte. I archived the old candidate implementation and original test source. The existing test assertions are unchanged; only their candidate-module import now points to the baseline implementation.

The wide pool has 750 candidates across all 50 episodes. Types: {'episode_ending': 429, 'adjacent_join': 199, 'single': 122}. There are 556 pool options without hook keywords, demonstrating that keywords no longer gate admission.

The full offline run made 145 mock stage calls: 50 A, 94 B and one C. It produced 79 fixture ranking rows and 671 rejection records, accounting for all 750 offered candidates. The mock cold-viewer pass is stored separately; cold_viewer_pass is null. These are not real model judgments or recommendations.

36 tests passed. Provider HTTP adapters were tested with injected Gemini/OpenAI-compatible response fixtures, not live services. ZIP and directory parsing/candidate outputs match exactly. Replaying all 145 cached responses reproduces ranked JSON, CSV and rejection CSV byte-for-byte. Cache-hit/cost log fields intentionally differ between fresh and replayed runs.

The Colab notebook has five code cells; each cell compiles. I did not execute a real Colab session or call a live provider. Its runtime-specific secret/upload/download steps therefore still need a live smoke test.

`verification.json` records these checks. `outputs_llm/tests.log` contains all test results. `cost_estimate.json` records the assumptions and arithmetic behind the README estimates.

Next step: run the notebook with an eligible Gemini key and inspect real-model failures before final submission.

Risks/failures to log:
- Conservative whole-event hook timing can reject useful hooks inside longer lines.
- Ending cues and separate model judgments can still misread resolved goals, rhetorical questions and hidden context.
- Free-tier quotas and large global ranking responses may interrupt a live run; cached successes remain available.
