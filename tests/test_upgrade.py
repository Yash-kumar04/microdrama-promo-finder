"""Regression, adversarial grounding, provider fixtures, retries and cache isolation."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from candidates import generate, evaluate, segment
from llm_layer import (A_KEYS, A_SYSTEM, B_SYSTEM, C_SYSTEM, ValidationError, validate_review,
                       validate_cold, normalise_review_row, validate_ranking, event_map, build_stage_a, pipeline, clip_text)
from llm_providers import Client, ProviderError, load_config, mock_response, parse_response

class WidePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'outputs/episodes.jsonl').exists():
            raise unittest.SkipTest('Generate outputs first to test the wide pool')
        cls.eps = [json.loads(x) for x in (ROOT/'outputs/episodes.jsonl').read_text().splitlines()]
        cls.rubric = json.loads((ROOT/'rubric.yaml').read_text())
        cls.pool, _, cls.stats = generate(cls.eps, cls.rubric)

    def test_fifteen_per_episode_and_ending_coverage(self):
        self.assertEqual(len(self.pool), sum(r['selected'] for r in self.stats['per_episode']))
        for row in self.stats['per_episode']:
            self.assertLessEqual(row['selected'], 15)
            self.assertEqual(sum(row['by_kind'].values()), row['selected'])
        # Keyword-free retention is checked independently in test_no_keyword_gate.

    def test_join_geometry(self):
        for c in self.pool:
            self.assertTrue(20 <= c['duration_sec'] <= 60)
            if c['kind'] == 'adjacent_join':
                a, b = c['segments']
                self.assertTrue(15-1e-6 <= a['end_sec']-a['start_sec'] <= 40+1e-6)
                self.assertTrue(5-1e-6 <= b['end_sec']-b['start_sec'] <= 20+1e-6)
                self.assertEqual(b['episode'], a['episode']+1)
                self.assertEqual(a['end_sec'], next(e for e in self.eps if e['number'] == a['episode'])['duration_sec'])
                self.assertEqual(b['start_sec'], 0)

    def test_no_keyword_gate(self):
        rubric = copy.deepcopy(self.rubric)
        rubric['heuristics']['hook_regex'] = r'NEVER_MATCH_ANY_SOURCE'
        if not self.pool:
            self.skipTest('No duration-valid candidates in this input')
        c = self.pool[0]
        r = evaluate(c['segments'], {e['number']: e for e in self.eps}, rubric, c['kind'])
        self.assertIsNotNone(r)
        self.assertEqual(r['score_components']['keyword_bonus'], 0)

    def test_deterministic_pool(self):
        again, _, stats = generate(self.eps, self.rubric)
        self.assertEqual(again, self.pool)
        self.assertEqual(stats, self.stats)

    def test_stage_b_only_clip_text(self):
        events = event_map(self.eps)
        if not self.pool:
            self.skipTest('No duration-valid candidates in this input')
        c = self.pool[0]
        text = clip_text(c['segments'], events)
        self.assertNotIn(self.eps[0]['logline'], text)
        self.assertNotIn(c['id'], text)
        self.assertNotIn('heuristic_score', text)

class ValidatorTests(unittest.TestCase):
    def setUp(self):
        # Explicit synthetic dialogue fixtures, never presented as source screenplay.
        es = []
        for i in range(20):
            es.append({'id': f'synthetic_{i}', 'episode': 1, 'scene': 1, 'type': 'dialogue',
                       'speaker': 'TEST', 'tone': 'test', 'start_sec': i*2, 'end_sec': (i+1)*2,
                       'raw_start': f'00:{i*2:05.2f}', 'raw_end': f'00:{(i+1)*2:05.2f}',
                       'text': f'Fixture sentence {i}.', 'source_order': i, 'source_line': i})
        self.ep = {'number': 1, 'duration_sec': 40, 'events': es, 'logline': 'SYNTHETIC LOG',
                   'continuity_notes': 'SYNTHETIC CONTINUITY', 'source_file': 'synthetic.txt',
                   'source_sha256': 'synthetic', 'scenes': [{'end_sec': 40}], 'audit': {'issues': []}}
        self.events = event_map([self.ep])
        self.c = {'id': 'candidate_test', 'owner_episode': 1, 'episodes': [1], 'kind': 'single',
                  'segments': [segment(self.ep, 0, 30)], 'duration_sec': 30}
        self.row = {'candidate_id': 'candidate_test', 'decision': 'accept', 'segments': [],
                    'hook_event_id': 'synthetic_0', 'hook_text': 'Fixture sentence 0.', 'hook_why': 'Test hook.',
                    'cliffhanger_event_id': 'synthetic_14', 'cliffhanger_text': 'Fixture sentence 14.',
                    'cliffhanger_why': 'Test end.', 'why_it_works': 'Test only.',
                    'scores': {'hook': 4, 'cliffhanger': 4, 'standalone': 4, 'intensity': 4},
                    'resolved_story_goal': False, 'rhetorical_question': False, 'reject_reason': ''}

    def validate(self, row=None):
        return validate_review(row or self.row, {'candidate_test': self.c}, {1: self.ep}, self.events, set(self.events))

    def test_valid_source_ids_and_custom_boundaries(self):
        p = self.validate()
        self.assertEqual(p['total_seconds'], 30)
        self.row['segments'] = [{'start_event_id': 'synthetic_0', 'end_event_id': 'synthetic_14'}]
        self.assertEqual(self.validate()['segments'], p['segments'])

    def test_reject_bad_ids_quotes_timestamps(self):
        for key, value in [('candidate_id', 'invented'), ('hook_event_id', 'invented'), ('hook_text', 'fabricated quotation')]:
            row = copy.deepcopy(self.row); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): self.validate(row)
        row = dict(self.row, start_sec=0)
        with self.assertRaises(ValidationError): self.validate(row)
        self.row['segments'] = [{'start_sec': 0, 'end_sec': 30}]
        with self.assertRaises(ValidationError): self.validate()

    def test_reject_late_hook(self):
        self.row['hook_event_id'] = 'synthetic_2'
        self.row['hook_text'] = self.events['synthetic_2']['text']
        with self.assertRaisesRegex(ValidationError, 'first_3s'): self.validate()

    def test_reject_long_hook_even_if_it_starts_in_first_three_seconds(self):
        self.events['synthetic_0']['end_sec'] = 3.5
        with self.assertRaisesRegex(ValidationError, 'first_3s'): self.validate()

    def test_reject_resolved_and_rhetorical(self):
        for key in ['resolved_story_goal', 'rhetorical_question']:
            row = dict(self.row); row[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, key): self.validate(row)

    def test_reject_duration_and_split_line(self):
        self.row['segments'] = [{'start_event_id': 'synthetic_0', 'end_event_id': 'synthetic_5'}]
        with self.assertRaisesRegex(ValidationError, 'duration'): self.validate()
        self.row['segments'] = [{'start_event_id': 'synthetic_1', 'end_event_id': 'synthetic_14'}]
        self.events['synthetic_0']['end_sec'] = 2.5
        with self.assertRaisesRegex(ValidationError, 'splits_line'): self.validate()

    def test_reject_quote_outside_clip_and_early_cliffhanger(self):
        for eid in ['synthetic_19', 'synthetic_9']:
            row = dict(self.row, cliffhanger_event_id=eid, cliffhanger_text=self.events[eid]['text'])
            with self.subTest(eid=eid), self.assertRaises(ValidationError): self.validate(row)

    def test_reject_bad_scores(self):
        for value in [-1, 6, True, float('nan'), '5']:
            row = copy.deepcopy(self.row); row['scores']['hook'] = value
            with self.subTest(value=value), self.assertRaises(ValidationError): self.validate(row)

    def test_cold_fail_and_schema(self):
        value = {'confusing': [], 'missing_context': [], 'would_keep_watching': 'Y', 'what_happens_next': 'Unknown.'}
        self.assertTrue(validate_cold(value))
        self.assertTrue(validate_cold(dict(value, missing_context=['Who is the bride?'])))   # one blocker tolerated
        self.assertFalse(validate_cold(dict(value, missing_context=['Who is the bride?', 'Who is the duke?'])))
        self.assertTrue(validate_cold(dict(value, would_keep_watching='yes', extra_key=1)))   # tolerant parsing
        self.assertFalse(validate_cold(dict(value, would_keep_watching='N')))
        with self.assertRaises(ValidationError): validate_cold(dict(value, would_keep_watching='maybe'))

    def test_ranker_requires_exact_permutation(self):
        ps = [{'promo_id': 'a'}, {'promo_id': 'b'}]
        good = {'ranked': [{'promo_id': 'b', 'reason': 'Test'}, {'promo_id': 'a', 'reason': 'Test'}], 'top_3': ['b', 'a']}
        self.assertEqual(validate_ranking(good, ps), good)
        for bad in [{'ranked': good['ranked'][:1], 'top_3': ['b']},
                    {'ranked': good['ranked'], 'top_3': ['a', 'b']},
                    {'ranked': [good['ranked'][0]]*2, 'top_3': ['b', 'b']}]:
            with self.assertRaises(ValidationError): validate_ranking(bad, ps)

    def test_pipeline_logs_omissions_and_uses_isolated_cold_prompt(self):
        cfg = load_config(ROOT/'config.yaml')
        class FixtureClient:
            calls = []
            def call(inner, stage, label, system, payload, validate=None):
                if stage == 'A': return {'reviews': [self.row]}
                if stage == 'B':
                    self.assertEqual(set(payload), {'clip_text'})
                    self.assertNotIn('SYNTHETIC LOG', payload['clip_text'])
                    return {'confusing': [], 'missing_context': [], 'would_keep_watching': 'Y', 'what_happens_next': 'Test prediction.'}
                return mock_response(stage, payload)
        with tempfile.TemporaryDirectory() as d:
            result, log = pipeline([self.ep], [self.c], cfg, d, FixtureClient())
            self.assertEqual(len(result['promos']), 1)
            self.assertIsNone(result['promos'][0]['cold_viewer_pass'])
            self.assertIn('MOCK', result['promos'][0]['status'])

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = load_config(ROOT/'config.yaml')
        self.cfg.update(provider='gemini', cache_dir=str(Path(self.tmp.name)/'cache'), api_key_env='KUKU_TEST_KEY', min_request_interval_seconds=0)
        self.env = patch.dict(os.environ, {'KUKU_TEST_KEY': 'test-secret-never-log'})
        self.env.start(); self.addCleanup(self.env.stop)

    def client(self, transport, sleep=lambda _: None):
        return Client(self.cfg, self.tmp.name, transport=transport, sleep=sleep)

    @staticmethod
    def gemini_response(text='{"ok":true}'):
        return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]}}],
                'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 20, 'thoughtsTokenCount': 5}}

    def test_gemini_transport_usage_and_cache(self):
        captured = []
        def transport(url, headers, body, timeout):
            captured.append(body)
            self.assertTrue(url.endswith(':generateContent'))
            self.assertEqual(headers['x-goog-api-key'], 'test-secret-never-log')
            self.assertEqual(body['generationConfig']['responseMimeType'], 'application/json')
            return self.gemini_response()
        c = self.client(transport)
        self.assertEqual(c.call('A', 'test', 'system', {'test': 1}), {'ok': True})
        self.assertEqual(c.call('A', 'test', 'system', {'test': 1}), {'ok': True})
        self.assertEqual(len(captured), 1)
        self.assertEqual(c.calls[0]['output_tokens'], 25)
        self.assertEqual(c.calls[1]['estimated_cost_usd'], 0)
        self.assertNotIn('test-secret-never-log', json.dumps(c.calls))
        for f in (Path(self.tmp.name)/'raw').glob('*.json'):
            self.assertNotIn('test-secret-never-log', f.read_text())

    def test_openai_compatible_shape(self):
        self.cfg.update(provider='openai_compatible', base_url='http://localhost:1234/v1', model='local-model', api_key_env='')
        def transport(url, headers, body, timeout):
            self.assertEqual(url, 'http://localhost:1234/v1/chat/completions')
            self.assertNotIn('Authorization', headers)
            self.assertEqual(body['messages'][0]['role'], 'system')
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}],
                    'usage': {'prompt_tokens': 50, 'completion_tokens': 10}}
        c = self.client(transport)
        self.assertEqual(c.call('B', 'test', 'sys', {}), {'ok': True})

    def test_rate_limit_retry_and_header(self):
        calls, sleeps = [], []
        def transport(*args):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError('url', 429, 'rate limit', {'Retry-After': '3'}, io.BytesIO(b''))
            return self.gemini_response()
        c = self.client(transport, sleeps.append)
        c.call('A', 'retry', 'sys', {})
        self.assertEqual(sleeps, [3.0])
        self.assertEqual(len(calls), 2)

    def test_daily_quota_does_not_hammer_or_fallback(self):
        calls = []
        def transport(*args):
            calls.append(1)
            raise urllib.error.HTTPError('url', 429, 'quota', {'Retry-After': '3600'}, io.BytesIO(b''))
        c = self.client(transport)
        for n in range(2):
            with self.assertRaises(ProviderError): c.call('A', str(n), 'sys', {'n': n})
        self.assertEqual(len(calls), 1)

    def test_malformed_json_retries_and_accounts_for_tokens(self):
        calls = []
        def transport(*args):
            calls.append(1)
            return self.gemini_response('not json' if len(calls) == 1 else '{"ok":true}')
        c = self.client(transport)
        c.call('C', 'json', 'sys', {})
        self.assertEqual(c.calls[0]['billable_input_tokens_estimate'], 200)
        self.assertEqual(c.calls[0]['billable_output_tokens_estimate'], 50)

    def test_schema_validation_is_retried_before_caching(self):
        calls = []
        def transport(*args):
            calls.append(1)
            return self.gemini_response('{"bad":true}' if len(calls)==1 else '{"ok":true}')
        def check(value):
            if set(value) != {'ok'}: raise ValidationError('bad schema')
        c = self.client(transport)
        self.assertEqual(c.call('A', 'schema', 's', {}, validate=check), {'ok': True})
        self.assertEqual(len(calls), 2)

    def test_cache_invalidation_model_and_payload(self):
        calls = []
        c = self.client(lambda *args: (calls.append(1) or self.gemini_response()))
        c.call('A', 'a', 's', {'v': 1})
        c.call('A', 'b', 's', {'v': 2})
        c.cfg['model'] = 'another-model'
        c.call('A', 'c', 's', {'v': 2})
        self.assertEqual(len(calls), 3)

    def test_missing_key_never_mock_fallback(self):
        self.cfg['api_key_env'] = 'KUKU_NONEXISTENT_TEST_KEY'
        with patch.dict(os.environ, {}, clear=True):
            c = self.client(lambda *args: self.fail('Network should not be called'))
            with self.assertRaises(ProviderError): c.call('A', 'no_key', 's', {})
        self.assertEqual(c.calls[0]['status'], 'failed')

    def test_truncated_response_and_nonfinite_json_rejected(self):
        self.cfg['max_attempts'] = 1
        response = self.gemini_response()
        response['candidates'][0]['finishReason'] = 'MAX_TOKENS'
        c = self.client(lambda *args: response)
        with self.assertRaises(ProviderError): c.call('A', 'truncated', 's', {})
        with self.assertRaises(ValueError): parse_response('{"score": NaN}')

if __name__ == '__main__':
    unittest.main()


class TolerantRowTests(unittest.TestCase):
    def test_missing_and_extra_keys(self):
        from llm_layer import A_KEYS
        row = {'candidate_id': 'c1', 'decision': 'reject', 'reject_reason': 'no hook', 'surprise': 1}
        out = normalise_review_row(row)
        self.assertEqual(set(out), A_KEYS)
        self.assertEqual(out['segments'], [])
    def test_cliffhanger_why_falls_back_to_why_it_works(self):
        out = normalise_review_row({'candidate_id': 'c1', 'decision': 'accept', 'why_it_works': 'because'})
        self.assertEqual(out['cliffhanger_why'], 'because')
