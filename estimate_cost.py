#!/usr/bin/env python3
"""Planning estimate from actual Stage A prompts; not a token counter or invoice."""
import argparse
import json
import math
from pathlib import Path
from llm_layer import A_SYSTEM, B_SYSTEM, build_stage_a, clip_text, event_map
from llm_providers import atomic_json, canonical, load_config

def estimate(episodes, pool, cfg):
    epmap = {e['number']: e for e in episodes}
    events = event_map(episodes)
    a_chars = sum(len(A_SYSTEM)+len(canonical(build_stage_a(ep, [c for c in pool if c['owner_episode']==ep['number']], epmap, events, cfg['max_promos_per_episode']))) for ep in episodes)
    # Conservative: maximum clip-text size per episode, for every allowed acceptance.
    b_chars = sum(cfg['max_promos_per_episode']*max((len(B_SYSTEM)+len(canonical({'clip_text': clip_text(c['segments'], events)})) for c in pool if c['owner_episode']==ep['number']), default=0) for ep in episodes)
    n = len(episodes)
    results = []
    for count in (50, 500):
        scale = count/n
        survivors = count*cfg['max_promos_per_episode']
        # 1,500 chars/survivor + 300/logline for the global ranker; all survivors included.
        c_chars = 1500*survivors+300*count+2000
        inputs = math.ceil((a_chars*scale+b_chars*scale+c_chars)/4*1.25)
        outputs = 6000*count+400*survivors+100*survivors
        price = cfg['pricing']
        cost = (inputs*price['input_usd_per_million']+outputs*price['output_usd_per_million'])/1e6
        results.append({'episodes': count, 'max_normal_calls': count+survivors+1,
                        'input_tokens_estimate': inputs, 'output_tokens_estimate': outputs,
                        'paid_equivalent_usd': round(cost, 4), 'with_25_percent_retry_contingency_usd': round(cost*1.25, 4),
                        'offline_mock_cost_usd': 0})
    return {'basis': 'Measured A payload chars; worst clip text for B; C size allowance. chars/4 with 25% input margin. Outputs budgeted at A=6000/episode, B=400/clip, C=100/promo. Not measured model usage.',
            'reference_model': cfg['pricing']['reference_model'], 'pricing': cfg['pricing'], 'source_episode_count': n,
            'measured_stage_A_chars': a_chars, 'stage_B_worst_case_chars': b_chars, 'estimates': results,
            'limits': ['Assumes the same candidate/event density and at most two Stage A acceptances per episode.',
                       'Free tier costs zero only while eligible and within quota; completion in one day is not guaranteed.',
                       '500-episode global ranking may require a larger model context/output budget; budget failures are logged, not silently truncated.',
                       'Reasoning tokens, retries, model price changes and verbose responses can increase costs.']} 

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=Path,default=Path('outputs/episodes.jsonl'))
    p.add_argument('--candidates',type=Path,default=Path('outputs/candidate_pool.jsonl'))
    p.add_argument('--config',type=Path,default=Path('config.yaml'))
    p.add_argument('--out',type=Path,default=Path('cost_estimate.json'))
    a=p.parse_args()
    rows=lambda path:[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    result=estimate(rows(a.episodes),rows(a.candidates),load_config(a.config))
    atomic_json(a.out,result)
    print(json.dumps(result['estimates'],indent=2))
if __name__=='__main__':main()
