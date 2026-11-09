"""Real-corpus regression tests plus small explicitly synthetic parser cases."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser import parse_episode, seconds
from candidates_baseline import beats, clean_boundary, generate, shared_fraction, ending_signal
import re

class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths=sorted((ROOT/'input').rglob('*.txt'))
        if not cls.paths:
            raise unittest.SkipTest('Original screenplay corpus is not bundled')
        cls.eps=[parse_episode(p) for p in cls.paths]
        if sorted(e['number'] for e in cls.eps) != list(range(1,51)):
            raise unittest.SkipTest('Original 50-episode corpus required for source regressions')
        cls.epmap={e['number']:e for e in cls.eps}
        cls.cfg=json.loads((ROOT/'rubric.yaml').read_text())
        cls.pool,cls.selected,*_=generate(cls.eps,cls.cfg)

    def test_episode_6_uppercase_regression(self):
        ep=self.epmap[6]
        self.assertEqual(ep['duration_sec'],115.0)
        self.assertEqual(len(ep['events']),97)
        self.assertEqual([(s['start_sec'],s['end_sec']) for s in ep['scenes']],[(0.0,33.7),(33.7,115.0)])
        self.assertEqual(ep['scenes'][1]['heading'],'INT. WOODEN CABIN – CONTINUOUS')
        self.assertEqual(ep['events'][0]['type'],'camera')
        self.assertEqual(ep['events'][-1]['text'],'Warm hearth crackles blending with gentle string music fading out.')
        dialogue=next(e for e in ep['events'] if e['type']=='dialogue')
        self.assertEqual((dialogue['speaker'],dialogue['tone'],dialogue['start_sec'],dialogue['end_sec']),('MYSTERIOUS BEARDED MAN','Solemn, Gentle',7.2,9.85))
        self.assertEqual(sum(i['code']=='uppercase_scene_field' for i in ep['audit']['issues']),6)
        self.assertFalse(any(i['severity']=='error' for i in ep['audit']['issues']))

    def test_source_coverage_and_runtime(self):
        self.assertEqual(sorted(self.epmap),list(range(1,51)))
        self.assertEqual(sum(len(e['events']) for e in self.eps),3727)
        self.assertAlmostEqual(sum(e['duration_sec'] for e in self.eps),6162.62)
        for ep in self.eps:
            a=ep['audit']
            self.assertEqual(a['nonempty_lines'],a['accounted_nonempty_lines'])
            self.assertEqual(a['event_header_count'],len(ep['events']))
            self.assertEqual(a['timestamp_pair_count'],len(ep['events']))
            self.assertEqual([e['start_sec'] for e in ep['events']],sorted(e['start_sec'] for e in ep['events']))

    def test_every_event_traces_to_original(self):
        for path,ep in zip(self.paths,self.eps):
            lines=path.read_text(encoding='utf-8-sig').splitlines()
            for e in ep['events']:
                self.assertEqual(lines[e['source_line']-1],e['raw_header'])
                self.assertEqual(e['start_sec'],seconds(e['raw_start']))
                self.assertEqual(e['end_sec'],seconds(e['raw_end']))
                self.assertEqual('\n'.join(lines[e['time_line']:e['source_end_line']]).strip(),e['text'])

    def test_episode_22_cross_scene_sfx_is_preserved(self):
        ep=self.epmap[22];e=next(e for e in ep['events'] if e['id']=='e022_v0023')
        self.assertEqual((e['type'],e['start_sec'],e['end_sec']),('sfx',41.5,42.8))
        self.assertTrue(any(i['code']=='event_outside_scene' and i['line']==102 for i in ep['audit']['issues']))

    def test_parenthetical_identity_not_lost(self):
        ep=self.epmap[37]
        e=next(e for e in ep['events'] if 'Rupert)' in (e['speaker'] or ''))
        self.assertEqual(e['speaker'],'BLOND NOBLEMAN (Rupert)')
        self.assertIn('Rupert',e['speaker_qualifiers'])

    def test_candidate_grounding_constraints_and_joins(self):
        self.assertTrue(self.selected)
        self.assertTrue(any(c['kind']=='adjacent_join' for c in self.selected))
        for c in self.pool:
            self.assertGreaterEqual(c['duration_sec'],20)
            self.assertLessEqual(c['duration_sec'],60)
            self.assertLessEqual(c['hook']['promo_onset_sec'],3)
            self.assertGreaterEqual(c['hook']['promo_onset_sec'],0)
            self.assertAlmostEqual(c['duration_sec'],sum(s['end_sec']-s['start_sec'] for s in c['segments']))
            for s in c['segments']:
                ep=self.epmap[s['episode']]
                self.assertTrue(clean_boundary(ep,s['start_sec']))
                self.assertTrue(clean_boundary(ep,s['end_sec']))
                self.assertIn(s['start_sec'],[e['start_sec'] for e in ep['events']])
                self.assertIn(s['end_sec'],[e['end_sec'] for e in ep['events']])
            for field in ('hook','cliffhanger'):
                v=c[field];e=next(e for e in self.epmap[v['episode']]['events'] if e['id']==v['event_id'])
                self.assertEqual(v['text'],e['text'])
            if len(c['segments'])==2:
                a,b=c['segments'];self.assertEqual(b['episode'],a['episode']+1)
                self.assertEqual(a['end_sec'],self.epmap[a['episode']]['duration_sec'])
                self.assertEqual(b['start_sec'],0)

    def test_global_overlap_suppression(self):
        for i,a in enumerate(self.selected):
            for b in self.selected[i+1:]: self.assertEqual(shared_fraction(a,b),0)
        for n in self.epmap: self.assertLessEqual(sum(n in c['episodes'] for c in self.selected),8)

    def test_reproducible_output(self):
        pool,selected,*_=generate(self.eps,self.cfg)
        self.assertEqual(selected,self.selected)
        self.assertEqual(pool,self.pool)

class SyntheticTests(unittest.TestCase):
    def parse(self,body):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episode-001.txt';p.write_text(body)
            return parse_episode(p)

    def test_bad_time_is_error_not_guessed(self):
        ep=self.parse('EPISODE NUMBER: Episode 1\nSCENE 01\n**CAMERA:**\nStart: (00:61.00) | End: (01:02.00)\nSynthetic test text.\n')
        self.assertTrue(any(i['code']=='invalid_event_time' for i in ep['audit']['issues']))
        self.assertEqual(ep['events'],[])

    def test_resolved_blacklist_is_not_cliffhanger(self):
        cfg=json.loads((ROOT/'rubric.yaml').read_text())
        regex={k.removesuffix('_regex'):re.compile(v,re.I) for k,v in cfg['heuristics'].items()}
        # Synthetic fixture, not screenplay dialogue.
        self.assertIsNone(ending_signal({'type':'dialogue','text':'"Everything is fine. Thank you?"'},0,[],regex))
        self.assertIsNone(ending_signal({'type':'dialogue','text':'"I have finished."'},0,[],regex))

    def test_overlapping_dialogue_cannot_be_cut(self):
        ep={'number':1,'events':[{'id':'synthetic_a','type':'dialogue','start_sec':0,'end_sec':5},
                                {'id':'synthetic_b','type':'action','start_sec':4,'end_sec':8},
                                {'id':'synthetic_c','type':'dialogue','start_sec':8,'end_sec':10}]}
        b=beats(ep)
        self.assertEqual([(x['start_sec'],x['end_sec']) for x in b],[(0,8),(8,10)])
        self.assertFalse(clean_boundary(ep,5));self.assertTrue(clean_boundary(ep,8))

if __name__=='__main__': unittest.main()
