#!/usr/bin/env python3
"""Run offline parsing + candidates on a ZIP or directory, without changing input."""
import argparse
import json
from pathlib import Path
import stat
import tempfile
import zipfile
from parser import parse_episode, audit_report, write_jsonl
from candidates import main as candidate_main
import sys

def unpack(archive,destination):
    with zipfile.ZipFile(archive) as z:
        members=[i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith('.txt') and '__MACOSX' not in Path(i.filename).parts]
        if len(members)>10000 or sum(i.file_size for i in members)>256*1024*1024:
            raise ValueError('Archive exceeds 10,000 texts or 256 MiB uncompressed')
        used=set()
        for i in members:
            name=Path(i.filename)
            if name.is_absolute() or name.drive or '..' in name.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16):
                raise ValueError(f'Unsafe archive member: {i.filename}')
            target=destination/name
            if target in used: raise ValueError(f'Duplicate archive member: {i.filename}')
            used.add(target);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(i))
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('input'));p.add_argument('--out',type=Path,default=Path('outputs'));p.add_argument('--rubric',type=Path,default=Path(__file__).with_name('rubric.yaml'))
    p.add_argument('--llm', action='store_true', help='Run A/B/C after deterministic generation')
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--provider', choices=['mock','gemini','openai_compatible'])
    p.add_argument('--llm-out', type=Path, default=Path('outputs_llm'))
    a=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder=unpack(a.input,Path(tmp)) if a.input.suffix.lower()=='.zip' else a.input
        paths=sorted(folder.rglob('*.txt')) if folder.is_dir() else [folder]
        if not paths: p.error('No .txt files found')
        eps=sorted((parse_episode(x) for x in paths),key=lambda e:(e['number'] or -1,e['source_file']))
        a.out.mkdir(parents=True,exist_ok=True);write_jsonl(a.out/'episodes.jsonl',eps);summary=audit_report(eps,a.out)
        print(json.dumps(summary,indent=2))
        fatal=[i for l in (a.out/'audit_issues.jsonl').read_text().splitlines() if (i:=json.loads(l))['severity']=='error' and i['code']!='event_outside_scene']
        if fatal: raise SystemExit('Structural parse errors found. Reports retained; candidate generation stopped.')
        if summary['issue_counts'].get('event_outside_scene'):
            print('NOTICE: cross-scene event timestamps preserved; candidates use event times, not clipped scene times.')
        original=sys.argv
        try:
            sys.argv=['candidates.py','--episodes',str(a.out/'episodes.jsonl'),'--out',str(a.out),'--rubric',str(a.rubric)]
            candidate_main()
        finally:sys.argv=original
        if a.llm:
            from llm_layer import pipeline
            from llm_providers import load_config
            cfg=load_config(a.config)
            if a.provider: cfg['provider']=a.provider
            pool=[json.loads(l) for l in (a.out/'candidate_pool.jsonl').read_text().splitlines() if l.strip()]
            _, log=pipeline(eps,pool,cfg,a.llm_out)
            print(json.dumps({k:log[k] for k in ('status','ranked','rejected_records','cache_hits')},indent=2))
            if log['failed_stages']: raise SystemExit('Review failures logged; inspect outputs before submission.')
if __name__=='__main__':main()
