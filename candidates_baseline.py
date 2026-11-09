#!/usr/bin/env python3
"""Offline, deterministic beat windows; scores are screening proxies, not promo approval."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from parser import stamp, write_jsonl

EPS=1e-6

def beats(ep):
    """Temporal components of overlapping dialogue/action. Touching lines stay separate.

    Camera/SFX are auxiliary tracks, not beat boundaries unless episode edges need
    them. Their full source events remain referenced; they do not glue a scene into
    one giant beat. No transcript word-level timing or semantic beats are invented.
    """
    result=[]
    for e in ep['events']:
        if e['type'] not in ('dialogue','action'): continue
        if result and e['start_sec']<result[-1]['end_sec']-EPS:
            result[-1]['end_sec']=max(result[-1]['end_sec'],e['end_sec'])
            result[-1]['event_ids'].append(e['id'])
        else:
            result.append({'episode':ep['number'],'start_sec':e['start_sec'],
                           'end_sec':e['end_sec'],'event_ids':[e['id']]})
    for i,b in enumerate(result): b['id']=f"e{ep['number']:03d}_b{i+1:03d}"
    return result

def clean_boundary(ep,t):
    return not any(e['type'] in ('dialogue','action') and e['start_sec']+EPS<t<e['end_sec']-EPS for e in ep['events'])

def boundary_refs(ep,t,which):
    return [e['id'] for e in ep['events'] if abs(e[which]-t)<EPS]

def segment(ep,start,end):
    if end<=start or not clean_boundary(ep,start) or not clean_boundary(ep,end): return None
    starts=boundary_refs(ep,start,'start_sec');ends=boundary_refs(ep,end,'end_sec')
    if not starts or not ends: return None
    ev=[e for e in ep['events'] if e['start_sec']<end-EPS and e['end_sec']>start+EPS]
    return {'episode':ep['number'],'start_sec':start,'end_sec':end,
            'start':next(e['raw_start'] for e in ep['events'] if e['id']==starts[0]),
            'end':next(e['raw_end'] for e in ep['events'] if e['id']==ends[0]), 'start_boundary_event_ids':starts,
            'end_boundary_event_ids':ends,'event_ids':[e['id'] for e in ev],
            'source_file':ep['source_file'],'source_sha256':ep['source_sha256']}

def speaker_key(s):
    # Strip delivery notation only; no show-specific alias guesses.
    return re.sub(r'\s*\((?:O\.S\.|V\.O\.|Off-screen)\)', '', s or '', flags=re.I).strip()

def interval_union(intervals):
    total=0;end=-1
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total

def ending_signal(last, end, previous, regex):
    text=last['text'].rstrip(' \n\"”\'’')
    if regex['resolved'].search(text): return None
    if last['type']=='dialogue' and text.endswith('?'): return 'question'
    # Ellipses can be hesitation, so retain only as a weak interruption signal.
    if last['type']=='dialogue' and text.endswith(('...','…','—','--','-')): return 'interruption_proxy'
    if last['type']=='action' and regex['interrupted_action'].search(text): return 'interrupted_action_proxy'
    if last['type']=='dialogue' and regex['reveal'].search(text):
        # The last semantic event must itself be the revelation; no later reaction.
        return 'reveal_without_following_semantic_event_proxy'
    return None

def evaluate(segs, epmap, cfg, regex, rejection):
    if any(s is None for s in segs): rejection['unsafe_boundary']+=1; return None
    duration=round(sum(s['end_sec']-s['start_sec'] for s in segs),6)
    hc=cfg['hard_constraints']
    if not hc['min_duration_sec']-EPS<=duration<=hc['max_duration_sec']+EPS:
        rejection['duration']+=1; return None
    items=[];offset=0
    for s in segs:
        ep=epmap[s['episode']]
        for e in ep['events']:
            if e['id'] in s['event_ids']:
                items.append((e,offset+e['start_sec']-s['start_sec'],offset+min(e['end_sec'],s['end_sec'])-s['start_sec']))
        offset+=s['end_sec']-s['start_sec']
    semantic=[x for x in items if x[0]['type'] in ('dialogue','action')]
    if not semantic: rejection['no_semantic_events']+=1; return None
    # Only text of newly starting semantic events is eligible, not camera prose or tone.
    hooks=[x for x in semantic if 0-EPS<=x[1]<=hc['hook_deadline_sec']+EPS and regex['hook'].search(x[0]['text'])]
    if not hooks: rejection['no_hook_proxy']+=1; return None
    last=max(semantic,key=lambda x:(x[2],x[1],x[0]['source_order']))
    if abs(last[2]-duration)>EPS: rejection['ending_not_semantic_boundary']+=1; return None
    ending=ending_signal(last[0],duration,semantic,regex)
    if not ending: rejection['no_unresolved_proxy']+=1; return None
    # Do not accept a question while another overlapping speaker/action completes it.
    tied=[x for x in semantic if abs(x[2]-duration)<EPS and x[0]['id']!=last[0]['id']]
    if tied: rejection['ambiguous_simultaneous_ending']+=1; return None
    hook=min(hooks,key=lambda x:(x[1],x[2],x[0]['id']))
    deadline=hc['hook_deadline_sec']
    dialogue_coverage=interval_union([(max(0,a),min(deadline,b)) for e,a,b in items if e['type']=='dialogue' and b>0 and a<deadline])/deadline
    speakers=sorted({speaker_key(e['speaker']) for e,a,b in semantic if e['type']=='dialogue'})
    # Upper-bound proxy only: speaker names are not necessarily known to a viewer.
    context_count=len(speakers)
    hook_hits=sorted({m.group(0).lower() for e,a,b in hooks for m in regex['hook'].finditer(e['text'])})
    final_ep=epmap[segs[-1]['episode']]; final_end=segs[-1]['end_sec']
    sf=any(abs(final_end-s['end_sec'])<EPS for s in final_ep['scenes'])
    ef=abs(final_end-final_ep['duration_sec'])<EPS
    emotions=sum(bool(regex['emotion'].search(e['text']+' '+(e['tone'] or ''))) for e,a,b in semantic)
    camera_boundaries=[]
    for s in segs:
        ep=epmap[s['episode']]
        for t in (s['start_sec'],s['end_sec']):
            camera_boundaries.append(not any(e['type']=='camera' and e['start_sec']+EPS<t<e['end_sec']-EPS for e in ep['events']))
    components={
      'hook':min(1,.55+.1*min(len(hook_hits),3)+.15*dialogue_coverage),
      'cliffhanger':{'question':.8,'interruption_proxy':.55,'interrupted_action_proxy':.7,'reveal_without_following_semantic_event_proxy':.65}[ending],
      'standalone':max(.1,1-.18*max(0,context_count-2)-.15*(len(segs)-1)),
      'emotional_intensity':min(1,emotions/4),
      'clean_cut':sum(camera_boundaries)/len(camera_boundaries)}
    score=sum(cfg['weights'][k]*v for k,v in components.items())
    landing=cfg['ranking']['landing_bonus_max']/segs[0]['episode']
    signature=json.dumps([(s['episode'],s['start_sec'],s['end_sec']) for s in segs],separators=(',',':'))
    flags=['hook_word_has_no_word_level_timestamp','cliffhanger_semantics_unverified','standalone_unverified']
    if len(segs)>1: flags.append('cross_episode_coherence_unverified')
    if ending=='interruption_proxy': flags.append('ellipsis_may_be_hesitation_not_interruption')
    if not all(camera_boundaries): flags.append('cut_inside_camera_shot')
    if any(epmap[s['episode']]['audit']['issues'] for s in segs): flags.append('source_audit_has_notices')
    def evidence(item):
        e,a,b=item
        return {'event_id':e['id'],'episode':e['episode'],'source_line':e['source_line'],
                'speaker':e['speaker'],'text':e['text'],'start_sec':e['start_sec'],'end_sec':e['end_sec'],
                'promo_onset_sec':round(a,6)}
    return {'id':'c_'+hashlib.sha256(signature.encode()).hexdigest()[:12],
      'kind':'single' if len(segs)==1 else 'adjacent_join','owner_episode':segs[0]['episode'],
      'episodes':[s['episode'] for s in segs],'segments':segs,'duration_sec':duration,
      'hook':evidence(hook),'cliffhanger':dict(evidence(last),signal=ending),
      'features':{'dialogue_density_first_3s':round(dialogue_coverage,6),'hook_keywords':hook_hits,
        'scene_final':sf,'episode_final':ef,'last_line_is_question':ending=='question',
        'last_line_is_interruption_proxy':'interruption' in ending or 'interrupted' in ending,
        'number_of_characters_needing_context_proxy':context_count,'speaker_labels':speakers,
        'emotional_event_count':emotions,'camera_boundary_fraction':components['clean_cut']},
      'component_scores':{k:round(v*100,3) for k,v in components.items()},
      'heuristic_score':round(score,6),'landing_bonus':round(landing,6),
      'base_rank_score':round(score+landing,6),'status':'needs_review','review_flags':flags,
      'why_it_might_work':f"Hook terms in an event starting at +{hook[1]:.2f}s; ends on {ending}; {context_count} speaker labels to understand. This is a screening explanation, not semantic approval."}

def shared_fraction(a,b):
    shared=sum(max(0,min(x['end_sec'],y['end_sec'])-max(x['start_sec'],y['start_sec'])) for x in a['segments'] for y in b['segments'] if x['episode']==y['episode'])
    return shared/min(a['duration_sec'],b['duration_sec'])

def select(pool,cfg):
    # Greedy diversity-aware ranking with global source-overlap suppression.
    remaining=list(pool); selected=[]; counts=Counter(); reject=Counter()
    while remaining:
        def adjusted(c): return c['base_rank_score']-cfg['ranking']['diversity_penalty_per_prior_episode']*sum(counts[n] for n in c['episodes'])
        best=min(remaining,key=lambda c:(-adjusted(c),c['id']));remaining.remove(best)
        if any(counts[n]>=cfg['ranking']['max_per_episode'] for n in best['episodes']): reject['episode_cap']+=1; continue
        if any(shared_fraction(best,s)>cfg['ranking']['max_overlap_fraction']+EPS for s in selected): reject['overlap']+=1; continue
        best=dict(best,rank=len(selected)+1,rank_score=round(adjusted(best),6),
                  diversity_penalty=round(best['base_rank_score']-adjusted(best),6))
        best['ranking_reason']='Weighted heuristic score + early landing bonus − repeated-episode penalty; source-overlap suppression applied.'
        selected.append(best);counts.update(best['episodes'])
    return selected,reject

def generate(eps,cfg):
    epmap={e['number']:e for e in eps}; bs={n:beats(e) for n,e in epmap.items()}
    regex={k.removesuffix('_regex'):re.compile(v,re.I) for k,v in cfg['heuristics'].items()}
    pool=[];rejection=Counter();attempts=Counter(); seen=set()
    fatal=[i for e in eps for i in e['audit']['issues'] if i['severity']=='error' and i['code']!='event_outside_scene']
    if fatal: raise ValueError('Structural parse errors must be resolved before candidate generation')
    def add(segs,kind):
        attempts[kind]+=1
        c=evaluate(segs,epmap,cfg,regex,rejection)
        if c and c['id'] not in seen: pool.append(c); seen.add(c['id'])
    lo=cfg['hard_constraints']['min_duration_sec'];hi=cfg['hard_constraints']['max_duration_sec']
    for n,ep in epmap.items():
        for i,a in enumerate(bs[n]):
            for b in bs[n][i:]:
                duration=b['end_sec']-a['start_sec']
                if duration>hi+EPS: break
                if duration>=lo-EPS: add([segment(ep,a['start_sec'],b['end_sec'])],'single')
        nxt=epmap.get(n+1)
        if not nxt: continue
        start_next=min(e['start_sec'] for e in nxt['events'])
        for a in bs[n]:
            tail=ep['duration_sec']-a['start_sec']
            if not cfg['generation']['min_join_component_sec']<=tail<=cfg['generation']['max_join_component_sec']: continue
            for b in bs[n+1]:
                head=b['end_sec']-start_next
                if head>cfg['generation']['max_join_component_sec']: break
                if head<cfg['generation']['min_join_component_sec'] or not lo-EPS<=tail+head<=hi+EPS: continue
                add([segment(ep,a['start_sec'],ep['duration_sec']),segment(nxt,start_next,b['end_sec'])],'adjacent_join')
    selected,suppression=select(pool,cfg)
    return pool,selected,bs,attempts,rejection,suppression

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=Path,default=Path('outputs/episodes.jsonl'));p.add_argument('--rubric',type=Path,default=Path(__file__).with_name('rubric.yaml'));p.add_argument('--out',type=Path,default=Path('outputs'));p.add_argument('--max-per-episode',type=int);p.add_argument('--max-overlap',type=float)
    a=p.parse_args();cfg=json.loads(a.rubric.read_text());eps=[json.loads(l) for l in a.episodes.read_text().splitlines() if l.strip()]
    if a.max_per_episode is not None: cfg['ranking']['max_per_episode']=a.max_per_episode
    if a.max_overlap is not None: cfg['ranking']['max_overlap_fraction']=a.max_overlap
    if not 0<=cfg['ranking']['max_overlap_fraction']<=1: p.error('--max-overlap must be in [0,1]')
    if cfg['ranking']['max_per_episode']<1: p.error('--max-per-episode must be positive')
    if len({e['number'] for e in eps})!=len(eps): p.error('Duplicate episode numbers')
    pool,selected,bs,attempts,rejection,suppression=generate(eps,cfg)
    a.out.mkdir(parents=True,exist_ok=True)
    write_jsonl(a.out/'candidate_pool.jsonl',sorted(pool,key=lambda c:(-c['base_rank_score'],c['id'])))
    write_jsonl(a.out/'candidates.jsonl',selected);write_jsonl(a.out/'beats.jsonl',[b for n in sorted(bs) for b in bs[n]])
    stats={'episodes':len(eps),'beats':sum(map(len,bs.values())),'windows_evaluated':dict(attempts),'rejections':dict(rejection),
           'eligible_pool':len(pool),'pool_by_kind':dict(Counter(c['kind'] for c in pool)),
           'selected':len(selected),'selected_by_kind':dict(Counter(c['kind'] for c in selected)),
           'suppression':dict(suppression),'per_episode':[], 'config':cfg}
    for ep in eps:
        n=ep['number'];stats['per_episode'].append({'episode':n,'pool_owned':sum(c['owner_episode']==n for c in pool),
                'selected_owned':sum(c['owner_episode']==n for c in selected),'selected_including_joins':sum(n in c['episodes'] for c in selected)})
    (a.out/'candidate_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    md=['# Candidate statistics','',f"I evaluated {sum(attempts.values()):,} windows: {dict(attempts)}. {len(pool)} passed screening; I retained {len(selected)} after ranking and overlap removal.",
        f"Retained types: {stats['selected_by_kind']}. Eligible types before suppression: {stats['pool_by_kind']}.",'',
        'These are unedited deterministic outputs, not approved promos. Every retained item needs semantic review. I enforce duration and whole dialogue/action boundaries exactly. Hook-event onset and ending patterns are proxies: the hook word may occur late in its line, a question may be rhetorical, and an ellipsis may be hesitation.',
        '',f"Overlap threshold: {cfg['ranking']['max_overlap_fraction']}; maximum selections touching one episode: {cfg['ranking']['max_per_episode']}. Default zero overlap removes all shared source footage, including adjacent joins. Eight is a ceiling; 70–201s episodes cannot supply eight disjoint clips of at least 20s each. I do not fill weak episodes to a quota.",
        '', 'I group strictly overlapping dialogue/action events into temporal beats; touching events stay separate. Camera and SFX do not merge beats. I enumerate 20–60s start/end beat pairs, and tails ending at the actual episode end joined to prefixes of the next numbered episode. Each join component is at least 3s and at most 50s. All segment boundaries match a parsed event boundary; camera/SFX may cross a cut. No line is shortened or rewritten.',
        '', 'The score uses the rubric weights. Character-context count is the number of speaker labels after removing delivery cues, not a verified count of unknown people. Scene/episode-final positions are logged but are not proof of a cliffhanger. Diversity is a 2-point penalty for each prior selection touching the same episode, plus strict overlap suppression. Landing bonus is 5 / first episode number, based only on my Ep 1 assumption.',
        '',f'Rejection reasons (first failing check): `{dict(rejection)}`.',f'Suppression reasons: `{dict(suppression)}`.','',
        '| Episode | Eligible owned | Retained owned | Retained touching episode |','|---:|---:|---:|---:|']
    for r in stats['per_episode']: md.append(f"| {r['episode']} | {r['pool_owned']} | {r['selected_owned']} | {r['selected_including_joins']} |")
    md+=['','## Run provenance','',f"episodes.jsonl SHA-256: `{hashlib.sha256(a.episodes.read_bytes()).hexdigest()}`.",
        'The exact effective rubric is stored in candidate_stats.json. No model, API call, key, network connection or paid dependency is used.']
    (a.out/'candidate_stats.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:v for k,v in stats.items() if k not in ('config','per_episode')},indent=2))
if __name__=='__main__':main()
