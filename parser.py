#!/usr/bin/env python3
"""Lossless text-screenplay parser and source-located mechanical audit (stdlib only)."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

TIME = r'\d{2,}:\d{2}(?:\.\d+)?'
PAIR = re.compile(rf'Start:\s*\(({TIME})\)\s*\|\s*End:\s*\(({TIME})\)', re.I)
HEADER = re.compile(r'\*\*(.+?):\*\*')
SCENE = re.compile(r'SCENE\s+(\d+)', re.I)
FIELD = re.compile(r'(EPISODE NUMBER|EPISODE LOGLINE|LOGLINE|CONTINUITY NOTES|SCENE HEADING|SCENE START|SCENE END):\s*(.*)', re.I)
KINDS = {'CAMERA': 'camera', 'ACTION': 'action', 'SOUND FX': 'sfx'}

def seconds(s):
    m, s = s.split(':')
    if not 0 <= float(s) < 60:
        raise ValueError('seconds component outside [0,60)')
    return round(int(m)*60 + float(s), 6)

def stamp(s):
    ticks = round(s * 100)
    return f'{ticks//6000:02d}:{(ticks%6000)/100:05.2f}'

def parse_episode(path):
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.decode('utf-8-sig').splitlines()
    ep = {'number': None, 'logline': '', 'continuity_notes': '', 'duration': None,
          'duration_sec': None, 'source_file': path.name,
          'source_sha256': hashlib.sha256(raw).hexdigest(), 'scenes': [], 'events': []}
    issues, consumed = [], set()
    scene = None
    def flag(code, line, detail, severity='warning'):
        issues.append({'episode': ep['number'], 'file': path.name, 'line': line,
                       'code': code, 'severity': severity, 'detail': detail})
    i = 0
    while i < len(lines):
        l = lines[i].strip(); loc = i+1
        if not l:
            consumed.add(i); i+=1; continue
        m = FIELD.fullmatch(l)
        if m:
            key, val = m.groups(); key = key.upper()
            if key == 'EPISODE NUMBER':
                n = re.fullmatch(r'(?:Episode\s*)?(\d+)', val, re.I)
                if not n: flag('invalid_episode_number',loc,l,'error')
                elif ep['number'] is not None: flag('duplicate_metadata',loc,l,'error')
                else: ep['number'] = int(n[1])
            elif key in ('EPISODE LOGLINE','LOGLINE','CONTINUITY NOTES'):
                dest = 'continuity_notes' if key == 'CONTINUITY NOTES' else 'logline'
                if ep[dest]: flag('duplicate_metadata',loc,l,'error')
                ep[dest] = val
            elif scene is None: flag('orphan_scene_field',loc,l,'error')
            else:
                dest = {'SCENE HEADING':'heading','SCENE START':'start_sec','SCENE END':'end_sec'}[key]
                if dest in scene: flag('duplicate_scene_field',loc,l,'error')
                if dest == 'heading': scene[dest]=val
                else:
                    try: scene[dest] = seconds(val)
                    except ValueError: flag('invalid_scene_time',loc,l,'error')
                scene.setdefault('raw_fields',{})[key] = val
                if l.split(':')[0].isupper(): flag('uppercase_scene_field',loc,l,'info')
            consumed.add(i); i+=1; continue
        m = SCENE.fullmatch(l)
        if m:
            scene = {'number':int(m[1]), 'source_line':loc}
            ep['scenes'].append(scene); consumed.add(i); i+=1; continue
        if l in ('--- SCRIPT START ---','--- SCRIPT END ---') or l.lower() == 'actions and dialogue:':
            if l == 'ACTIONS AND DIALOGUE:': flag('uppercase_section_label',loc,l,'info')
            consumed.add(i); i+=1; continue
        m = HEADER.fullmatch(l)
        if m:
            label = m[1]; header_i=i; consumed.add(i); i+=1
            while i < len(lines) and not lines[i].strip(): consumed.add(i); i+=1
            t = PAIR.fullmatch(lines[i].strip()) if i<len(lines) else None
            if not t:
                flag('missing_or_invalid_event_time',loc,label,'error'); continue
            try: start,end = map(seconds,t.groups())
            except ValueError:
                flag('invalid_event_time',i+1,lines[i],'error'); consumed.add(i); i+=1; continue
            time_line=i+1; consumed.add(i); i+=1; text_lines=[]
            while i<len(lines):
                s=lines[i].strip()
                if HEADER.fullmatch(s) or SCENE.fullmatch(s) or FIELD.fullmatch(s) or s.startswith('--- SCRIPT') or s.lower()=='actions and dialogue:': break
                if s.startswith('**'):
                    flag('malformed_event_header',i+1,s,'error')
                if PAIR.fullmatch(s): flag('extra_event_time',i+1,s,'error')
                text_lines.append(lines[i]); consumed.add(i); i+=1
            text='\n'.join(text_lines).strip()
            kind=KINDS.get(label.upper(),'dialogue'); speaker=tone=None; qualifiers=[]
            if kind=='dialogue':
                # Last parenthesis is tone; earlier groups stay in speaker for lossless identity.
                sm=re.fullmatch(r'(.*?)\s*\(([^()]*)\)',label)
                if sm: speaker,tone=sm.groups(); qualifiers=re.findall(r'\(([^()]*)\)',speaker)
                else: speaker=label; flag('speaker_without_tone',loc,label,'info')
                if qualifiers: flag('speaker_extra_parentheses',loc,label,'info')
                if re.search(r'\b(?:O\.S\.|V\.O\.)',tone or ''): flag('voice_cue_in_tone',loc,label,'info')
                if not (text.startswith(('"','“')) and text.endswith(('"','”'))): flag('dialogue_quote_variant',loc,text,'info')
            event={'episode':ep['number'],'scene':scene['number'] if scene else None,'type':kind,
                   'speaker':speaker,'tone':tone,'start_sec':start,'end_sec':end,'text':text,
                   'id':f'e{ep["number"]:03d}_v{len(ep["events"])+1:04d}' if ep['number'] is not None else f'unknown_{loc}',
                   'source_line':loc,'source_end_line':i,'time_line':time_line,
                   'source_order':len(ep['events']), 'raw_header':lines[header_i],
                   'raw_start':t[1],'raw_end':t[2], 'speaker_qualifiers':qualifiers,
                   'delivery_cues':re.findall(r'O\.S\.|V\.O\.|Off-screen',label,re.I)}
            ep['events'].append(event)
            if scene is None: flag('event_without_scene',loc,label,'error')
            if start>=end: flag('nonpositive_event_duration',loc,f'{start} → {end}','error')
            if not text: flag('empty_event',loc,label,'error')
            continue
        flag('unparsed_line',loc,l,'error'); i+=1
    for key in ('number','logline','continuity_notes'):
        if ep[key] is None or ep[key]=='': flag('missing_metadata',1,key,'error')
    for sentinel in ('--- SCRIPT START ---','--- SCRIPT END ---'):
        if sum(l.strip()==sentinel for l in lines)!=1: flag('script_marker_count',1,sentinel,'error')
    file_n=re.search(r'episode-(\d+)',path.name,re.I)
    if file_n and int(file_n[1]) != ep['number']: flag('filename_number_mismatch',1,path.name,'error')
    scenes=ep['scenes']; events=ep['events']
    if not scenes: flag('missing_scenes',1,'No scenes','error')
    if not events: flag('missing_events',1,'No events','error')
    if [s['number'] for s in scenes]!=list(range(1,len(scenes)+1)): flag('scene_number_sequence',1,str([s['number'] for s in scenes]),'error')
    for j,s in enumerate(scenes):
        if any(k not in s for k in ('heading','start_sec','end_sec')):
            flag('incomplete_scene',s['source_line'],str(s),'error'); continue
        if s['start_sec']>=s['end_sec']: flag('nonpositive_scene_duration',s['source_line'],str(s),'error')
        if j==0 and s['start_sec']!=0: flag('nonzero_episode_start',s['source_line'],str(s['start_sec']))
        if j and 'end_sec' in scenes[j-1] and s['start_sec']!=scenes[j-1]['end_sec']:
            flag('scene_gap_or_overlap',s['source_line'],f"Previous ends {scenes[j-1]['end_sec']}; starts {s['start_sec']}")
        evs=[e for e in events if e['scene']==s['number']]
        if not evs: flag('empty_scene',s['source_line'],str(s['number']),'error')
        for e in evs:
            if e['start_sec']<s['start_sec'] or e['end_sec']>s['end_sec']:
                flag('event_outside_scene',e['source_line'],f"{e['id']} {e['start_sec']}–{e['end_sec']}; scene {s['start_sec']}–{s['end_sec']}",'error')
    for a,b in zip(events,events[1:]):
        if b['start_sec']<a['start_sec']: flag('source_order_time_regression',b['source_line'],f"{a['id']} starts {a['start_sec']}; {b['id']} starts {b['start_sec']}",'info')
    ordered=sorted(events,key=lambda e:(e['start_sec'],e['source_order']))
    overlap_counts=Counter()
    for j,b in enumerate(ordered):
        for a in ordered[:j]:
            overlap=min(a['end_sec'],b['end_sec'])-max(a['start_sec'],b['start_sec'])
            if overlap>1e-6:
                pair='/'.join(sorted((a['type'],b['type']))); overlap_counts[pair]+=1
                if a['type']==b['type']:
                    flag('same_track_overlap',b['source_line'],f"{pair}: {a['id']} / {b['id']}: {overlap:.3f}s",'warning')
    seen={}
    for e in ordered:
        sig=tuple(e[k] for k in ('scene','type','speaker','tone','start_sec','end_sec','text'))
        if sig in seen: flag('duplicate_event',e['source_line'],f"{seen[sig]} / {e['id']}",'error')
        seen[sig]=e['id']
    end=max([s.get('end_sec',0) for s in scenes]+[e['end_sec'] for e in events]+[0])
    ep['duration']=ep['duration_sec']=end
    ep['duration_basis']='max declared scene end / event end; episode clock origin 0'
    ep['events']=ordered
    # Timeline gaps concern all tracks, not silence between dialogue lines.
    cursor=0
    for e in ordered:
        if e['start_sec']>cursor+1e-6: flag('uncovered_timeline_gap',e['source_line'],f'{cursor:.3f}–{e["start_sec"]:.3f}s')
        cursor=max(cursor,e['end_sec'])
    if end>cursor: flag('uncovered_timeline_tail',len(lines),f'{cursor:.3f}–{end:.3f}s')
    ep['audit']={'issues':issues,'overlap_pairs':dict(overlap_counts),
                 'nonempty_lines':sum(bool(l.strip()) for l in lines),
                 'accounted_nonempty_lines':sum(bool(lines[i].strip()) for i in consumed),
                 'event_header_count':sum(bool(HEADER.fullmatch(l.strip())) for l in lines),
                 'timestamp_pair_count':sum(bool(PAIR.fullmatch(l.strip())) for l in lines)}
    return ep

def write_jsonl(path, rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')

def audit_report(eps, out):
    issues=[i for ep in eps for i in ep['audit']['issues']]
    nums=[ep['number'] for ep in eps]
    for n,c in Counter(nums).items():
        if c>1: issues.append({'episode':n,'file':'(collection)','line':0,'code':'duplicate_episode','severity':'error','detail':f'{c} files'})
    valid=[n for n in nums if n is not None]
    missing=sorted(set(range(min(valid),max(valid)+1))-set(valid)) if valid else []
    for n in missing: issues.append({'episode':n,'file':'(collection)','line':0,'code':'missing_episode','severity':'error','detail':'Gap within supplied episode range'})
    write_jsonl(out/'audit_issues.jsonl',issues)
    durations=[e['duration_sec'] for e in eps]; counts=Counter(i['code'] for i in issues)
    summary={'episode_count':len(eps),'episode_numbers':nums,'total_runtime_sec':round(sum(durations),3),
             'min_duration_sec':min(durations),'median_duration_sec':statistics.median(durations),'max_duration_sec':max(durations),
             'scene_count':sum(len(e['scenes']) for e in eps),'event_count':sum(len(e['events']) for e in eps),
             'event_type_counts':dict(Counter(v['type'] for e in eps for v in e['events'])),
             'issue_counts':dict(counts),'severity_counts':dict(Counter(i['severity'] for i in issues)),
             'all_nonempty_lines_accounted':all(e['audit']['nonempty_lines']==e['audit']['accounted_nonempty_lines'] for e in eps),
             'headers_equal_events':all(e['audit']['event_header_count']==len(e['events'])==e['audit']['timestamp_pair_count'] for e in eps)}
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    md=['# Parse report','',f"I parsed {len(eps)} episodes, {summary['scene_count']} scenes and {summary['event_count']} events.",
        f"Total source-clock runtime: {sum(durations):.2f}s ({sum(durations)/60:.2f} min). Min / median / max: {min(durations):.2f} / {statistics.median(durations):.3f} / {max(durations):.2f}s.",'',
        'I use the maximum declared scene/event end per episode, measured from zero. I do not sum overlapping event lengths. These are screenplay timings, not measured video lengths.',
        '', '## Structure and normalization', '',
        '`EPISODE NUMBER: Episode N`, `EPISODE LOGLINE`, `CONTINUITY NOTES`, SCRIPT START marker, then SCENE blocks with heading/start/end, Actions and dialogue, bold event headers, timestamp pair and text; SCRIPT END marker.',
        'Scene field matching is case-insensitive. CAMERA/ACTION/SOUND FX become camera/action/sfx; other bold headers become dialogue. The last parenthesis is tone; earlier parentheticals remain in speaker and speaker_qualifiers. Text, raw headers, raw times, line references, source order and source SHA-256 are retained. Events are sorted by start time, stably for ties. Scene numbers are episode-local.',
        '',f"All nonempty lines accounted: {summary['all_nonempty_lines_accounted']}. Headers = timestamps = events for every file: {summary['headers_equal_events']}.",
        '', '## Every detected mechanical exception', '',
        'The summary below groups occurrences; the appendix and audit_issues.jsonl list every detected location. Info entries are format variations, not necessarily errors. Simultaneous camera/action/dialogue/SFX are expected; same-track overlaps need review, not automatic correction.',
        '', '| Code | Count | Episodes |','|---|---:|---|']
    for code,c in sorted(counts.items()):
        md.append(f"| {code} | {c} | {', '.join(map(str,sorted({i['episode'] for i in issues if i['code']==code},key=lambda n:n or -1)))} |")
    md+=['','## Episode inventory','','| Ep | Duration (s) | Scenes | Events | Issues |','|---:|---:|---:|---:|---:|']
    for e in eps: md.append(f"| {e['number']} | {e['duration_sec']:.2f} | {len(e['scenes'])} | {len(e['events'])} | {len(e['audit']['issues'])} |")
    md+=['','## Audit scope and limits','',
         'I check numbering, filename/header agreement, required metadata and script markers, unconsumed lines, event/header/time coverage, invalid or missing timestamps, nonpositive lengths, duplicate events, scene containment, scene gaps/overlaps, source-order regressions, same-track overlaps and uncovered timeline intervals. The audit never repairs timestamps.',
         'I do not claim to have detected every plot inconsistency. Speaker aliases, continuity-note truth, scene geography, and whether a revelation is resolved need semantic review. Speaker labels are not silently merged. A role label is not proof of a distinct person. See speaker_inventory.json for every observed label and episode.',
         '', '## Full exception appendix','']
    for i in issues: md.append(f"- Ep {i['episode']}, `{i['file']}:{i['line']}` — **{i['code']}** ({i['severity']}): {i['detail']}")
    (out/'parse_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    speakers=defaultdict(set)
    for ep in eps:
        for e in ep['events']:
            if e['speaker']: speakers[e['speaker']].add(ep['number'])
    (out/'speaker_inventory.json').write_text(json.dumps({k:sorted(v) for k,v in sorted(speakers.items())},indent=2)+'\n')
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--input',type=Path,default=Path('input'));ap.add_argument('--out',type=Path,default=Path('outputs'));ap.add_argument('--allow-errors',action='store_true')
    a=ap.parse_args();paths=sorted(a.input.rglob('*.txt')) if a.input.is_dir() else [a.input]
    if not paths: ap.error('No .txt screenplay files found')
    eps=sorted((parse_episode(p) for p in paths),key=lambda e:(e['number'] or -1,e['source_file']))
    a.out.mkdir(parents=True,exist_ok=True);write_jsonl(a.out/'episodes.jsonl',eps);summary=audit_report(eps,a.out)
    print(json.dumps(summary,indent=2))
    if summary['severity_counts'].get('error',0) and not a.allow_errors: sys.exit('Audit errors found; outputs retained; inspect audit_issues.jsonl before proceeding.')
if __name__=='__main__': main()
