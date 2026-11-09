#!/usr/bin/env python3
"""Wide deterministic recall pool. Keywords and ending cues only ADD points."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from candidates_baseline import beats, clean_boundary, segment, shared_fraction, interval_union, speaker_key
from parser import write_jsonl

EPS = 1e-6
ENDING_CUE = re.compile(r'cliffhanger|cut(?:s)? to black|abrupt(?:ly)? cut|\bholds?\b|\bfad(?:e|es|ing)\b', re.I)

def evaluate(segments, epmap, rubric, kind):
    if any(s is None for s in segments):
        return None
    duration = round(sum(s['end_sec'] - s['start_sec'] for s in segments), 6)
    if not 20 - EPS <= duration <= 60 + EPS:
        return None
    items, offset = [], 0
    for s in segments:
        for e in epmap[s['episode']]['events']:
            if e['id'] in s['event_ids']:
                items.append((e, offset + e['start_sec'] - s['start_sec'],
                              offset + min(e['end_sec'], s['end_sec']) - s['start_sec']))
        offset += s['end_sec'] - s['start_sec']
    hook_re = re.compile(rubric['heuristics']['hook_regex'], re.I)
    emotion_re = re.compile(rubric['heuristics']['emotion_regex'], re.I)
    early = [e for e, a, b in items if 0 <= a <= 3 and e['type'] in ('dialogue', 'action')]
    hits = sorted({m[0].lower() for e in early for m in hook_re.finditer(e['text'])})
    dialogue_density = interval_union([(max(0, a), min(3, b)) for e, a, b in items
                                       if e['type'] == 'dialogue' and a < 3 and b > 0]) / 3
    semantic = [(e, a, b) for e, a, b in items if e['type'] in ('dialogue', 'action')]
    final_line = max(semantic, key=lambda x: (x[2], x[1]), default=(None, 0, 0))[0]
    question = bool(final_line and final_line['text'].rstrip(' "”\'’').endswith('?'))
    interruption = bool(final_line and final_line['text'].rstrip(' "”\'’').endswith(('...', '…', '—', '--')))
    # ONLY final camera/action events of the final segment, not a cue elsewhere.
    visual = [(e, a, b) for e, a, b in items if e['type'] in ('camera', 'action')
              and b >= duration - 3 and a < duration]
    final_visual = max(visual, key=lambda x: (x[2], x[1]), default=(None, 0, 0))[0]
    cues = sorted(set(ENDING_CUE.findall(final_visual['text']))) if final_visual else []
    speakers = sorted({speaker_key(e['speaker']) for e, _, _ in items if e['speaker']})
    intensity = sum(bool(emotion_re.search(e['text'] + ' ' + (e['tone'] or ''))) for e, _, _ in semantic)
    ef = abs(segments[-1]['end_sec'] - epmap[segments[-1]['episode']]['duration_sec']) < EPS
    sf = any(abs(segments[-1]['end_sec'] - s['end_sec']) < EPS
             for s in epmap[segments[-1]['episode']]['scenes'])
    components = {'keyword_bonus': min(18, 6 * len(hits)), 'dialogue_opening': 8 * dialogue_density,
                  'ending_cue_bonus': 25 if cues else 0, 'episode_ending_bonus': 12 if ef else 0,
                  'question_bonus': 6 if question else 0, 'interruption_bonus': 4 if interruption else 0,
                  'intensity': min(10, 2 * intensity), 'few_speakers': max(0, 10 - 2 * max(0, len(speakers)-2)),
                  'landing_bonus': 5 / segments[0]['episode']}
    signature = json.dumps([(s['episode'], s['start_sec'], s['end_sec']) for s in segments], separators=(',', ':'))
    return {'id': 'c_' + hashlib.sha256(signature.encode()).hexdigest()[:12],
            'owner_episode': segments[0]['episode'], 'episodes': [s['episode'] for s in segments],
            'kind': kind, 'segments': segments, 'duration_sec': duration,
            'heuristic_score': round(sum(components.values()), 6), 'score_components': components,
            'features': {'hook_keywords': hits, 'dialogue_density_first_3s': round(dialogue_density, 6),
                         'last_line_is_question': question, 'last_line_is_interruption': interruption,
                         'final_visual_cues': cues, 'final_visual_event_id': final_visual['id'] if final_visual else None,
                         'episode_final': ef, 'scene_final': sf, 'speaker_count_proxy': len(speakers)},
            'status': 'unreviewed_wide_pool'}

def shortlist(pool, count=15):
    """Reserve ending/join options, then overlap-penalized fill. Never discard on keywords."""
    selected = []
    def choose(rows, target):
        rows = [c for c in rows if c not in selected]
        for _ in range(min(target, len(rows))):
            def key(c):
                overlap = max((shared_fraction(c, p) for p in selected), default=0)
                return (-(c['heuristic_score'] - 25 * overlap), c['id'])
            best = min(rows, key=key)
            selected.append(best)
            rows.remove(best)
    choose([c for c in pool if c['kind'] == 'episode_ending'], min(4, count))
    choose([c for c in pool if c['kind'] == 'adjacent_join'], min(4, max(0, count-len(selected))))
    choose(pool, count-len(selected))
    return [dict(c, pool_rank=i+1) for i, c in enumerate(selected)]

def generate(eps, rubric, per_episode=15):
    epmap = {e['number']: e for e in eps}
    if len(epmap) != len(eps):
        raise ValueError('Duplicate episode numbers')
    fatal = [i for e in eps for i in e['audit']['issues']
             if i['severity'] == 'error' and i['code'] != 'event_outside_scene']
    if fatal:
        raise ValueError('Structural parse errors; inspect parser audit')
    beatmap = {n: beats(e) for n, e in epmap.items()}
    stats, all_selected, all_counts = Counter(), [], []
    for n, ep in sorted(epmap.items()):
        windows = {}
        def add(segs, kind):
            stats['attempted_' + kind] += 1
            c = evaluate(segs, epmap, rubric, kind)
            if c is None:
                stats['invalid_boundary_or_duration'] += 1
                return
            # Episode-ending classification takes precedence over the same single window.
            if c['id'] not in windows or kind == 'episode_ending':
                windows[c['id']] = c
        bs = beatmap[n]
        for i, start in enumerate(bs):
            for end in bs[i:]:
                length = end['end_sec'] - start['start_sec']
                if length > 60 + EPS:
                    break
                if length >= 20 - EPS:
                    add([segment(ep, start['start_sec'], end['end_sec'])], 'single')
        # All event starts are considered for tail windows (not only semantic beats).
        starts = sorted({e['start_sec'] for e in ep['events'] if clean_boundary(ep, e['start_sec'])})
        for start in starts:
            if 20-EPS <= ep['duration_sec']-start <= 60+EPS:
                add([segment(ep, start, ep['duration_sec'])], 'episode_ending')
        nxt = epmap.get(n+1)
        if nxt:
            opening = min(e['start_sec'] for e in nxt['events'])
            ends = sorted({e['end_sec'] for e in nxt['events'] if clean_boundary(nxt, e['end_sec'])})
            for start in starts:
                tail = ep['duration_sec']-start
                if not 15-EPS <= tail <= 40+EPS:
                    continue
                for end in ends:
                    if 5-EPS <= end-opening <= 20+EPS:
                        add([segment(ep, start, ep['duration_sec']), segment(nxt, opening, end)], 'adjacent_join')
        pool = list(windows.values())
        selected = shortlist(pool, per_episode)
        all_selected.extend(selected)
        all_counts.append({'episode': n, 'unique_windows': len(pool), 'selected': len(selected),
                           'by_kind': dict(Counter(c['kind'] for c in selected)),
                           'no_keyword_options': sum(not c['features']['hook_keywords'] for c in selected),
                           'coverage_note': '' if selected else 'No valid 20–60s cuts; no timestamps fabricated.'})
    return all_selected, beatmap, {'counts': dict(stats), 'per_episode': all_counts,
                                  'selected': len(all_selected), 'per_episode_target': per_episode}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=Path, default=Path('outputs/episodes.jsonl'))
    p.add_argument('--rubric', type=Path, default=Path(__file__).with_name('rubric.yaml'))
    p.add_argument('--out', type=Path, default=Path('outputs'))
    p.add_argument('--per-episode', type=int, default=15)
    a = p.parse_args()
    if a.per_episode < 1:
        p.error('--per-episode must be positive')
    eps = [json.loads(l) for l in a.episodes.read_text().splitlines() if l.strip()]
    selected, bs, stats = generate(eps, json.loads(a.rubric.read_text()), a.per_episode)
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out/'candidate_pool.jsonl', selected)
    write_jsonl(a.out/'candidates.jsonl', selected)
    write_jsonl(a.out/'beats.jsonl', [b for n in sorted(bs) for b in bs[n]])
    (a.out/'candidate_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    md = ['# Wide candidate pool', '', f"I retained {len(selected)} candidates across {len(eps)} episodes.",
          'Keywords only add points. No hook, question, punctuation or resolution keyword rejects a window. The model decides semantic quality.',
          'I reserve up to four episode endings and four adjacent joins per episode, then fill to 15 using soft overlap penalties. Overlap is intentional at this recall stage.',
          'Endings: last 20–60s at source boundaries. Joins: last 15–40s plus first 5–20s. Final visual cues add 25 points; this is a prior, not evidence of an unresolved plot.', '',
          '| Episode | Valid windows | Pool | No keyword | Kinds |', '|---:|---:|---:|---:|---|']
    for r in stats['per_episode']:
        md.append(f"| {r['episode']} | {r['unique_windows']} | {r['selected']} | {r['no_keyword_options']} | {r['by_kind']} |")
    (a.out/'candidate_stats.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'selected': len(selected), 'episodes_with_options': sum(bool(x['selected']) for x in stats['per_episode']), 'counts': stats['counts']}, indent=2))

if __name__ == '__main__':
    main()
