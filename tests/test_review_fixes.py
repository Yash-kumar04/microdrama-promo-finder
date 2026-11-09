import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import test_upgrade as fixtures
from llm_layer import ValidationError, validate_cold, pipeline
from llm_providers import Client, ProviderError, load_config

class ReviewFixTests(unittest.TestCase):
    def fixture(self):
        f = fixtures.ValidatorTests(); f.setUp(); return f

    def test_required_accept_fields_and_timestamp_rejection(self):
        f = self.fixture()
        for key in ('resolved_story_goal', 'rhetorical_question', 'scores', 'hook_why', 'segments'):
            row = copy.deepcopy(f.row); del row[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, 'missing_required_accept_fields:.*'+key):
                f.validate(row)
        for key in ('cliffhanger_why', 'reject_reason'):
            row = copy.deepcopy(f.row); del row[key]
            self.assertIsNotNone(f.validate(row))
        row = dict(f.row, start_sec=0)
        with self.assertRaises(ValidationError): f.validate(row)
        row = dict(f.row, segments=[{'start_sec':0, 'end_sec':30}])
        with self.assertRaises(ValidationError): f.validate(row)

    def test_missing_flag_is_logged(self):
        f=self.fixture(); row=dict(f.row); del row['resolved_story_goal']
        class Replies:
            calls=[]
            def call(self, stage, *args, **kwargs):
                assert stage == 'A'
                return {'reviews':[row]}
        with tempfile.TemporaryDirectory() as d:
            _,log=pipeline([f.ep],[f.c],load_config(Path('config.yaml')),Path(d),client=Replies())
            self.assertEqual(log['A_validated'],0)
            self.assertIn('missing_required_accept_fields:resolved_story_goal',Path(d,'rejected_candidates.csv').read_text())

    def test_cold_required_fields_and_tolerance(self):
        row={'would_keep_watching':True,'confusing':None,'missing_context':None,'what_happens_next':'Unknown','extra':1}
        self.assertTrue(validate_cold(copy.deepcopy(row)))
        for key in ('would_keep_watching','confusing','missing_context','what_happens_next'):
            broken=dict(row); del broken[key]
            with self.subTest(key=key),self.assertRaisesRegex(ValidationError,'missing_required_B_fields:.*'+key):validate_cold(broken)
        self.assertTrue(validate_cold(dict(row,would_keep_watching='Yes',missing_context=['one'],confusing=['issue'])))
        self.assertFalse(validate_cold(dict(row,missing_context=['one','two'])))

    def test_provider_nonobject_retries(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'GEMINI_API_KEY':'offline-test'}):
            cfg=load_config(Path('config.yaml'));cfg.update(provider='gemini',cache_dir=d+'/cache',max_attempts=2,min_request_interval_seconds=0)
            calls=[]
            def transport(*args):calls.append(1);return []
            c=Client(cfg,Path(d),transport=transport,sleep=lambda _:None)
            with self.assertRaisesRegex(ProviderError,'invalid_provider_response'):c.call('A','invalid','test',{})
            self.assertEqual(len(calls),2)
            self.assertEqual(c.calls[0]['error'],'invalid_provider_response_ValueError')

    def test_empty_postprocess(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'promos_ranked.json').write_text('{"promos":[]}')
            r=subprocess.run([sys.executable,'postprocess_diverse.py',d],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('No promos survived',r.stdout)
            for name in ('promos_final_diverse.csv','promos_dropped_as_overlap.csv'):
                with Path(d,name).open() as f:self.assertEqual(len(list(csv.reader(f))),1)
