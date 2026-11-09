#!/usr/bin/env python3
"""Inventory formatting and a configurable name prefix; never normalizes story names."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from parser import PAIR,write_jsonl

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('input'));p.add_argument('--out',type=Path,default=Path('outputs'));p.add_argument('--name-prefix',default='Ash',help='Corpus-specific inspection aid, not an entity resolver')
    a=p.parse_args();forms=defaultdict(set);precision=Counter();names=[];speech=[]
    for path in sorted(a.input.rglob('*.txt')):
        lines=path.read_text(encoding='utf-8-sig').splitlines()
        ep=int(re.search(r'EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)', '\n'.join(lines),re.I)[1])
        for no,line in enumerate(lines,1):
            if re.match(r'(?:Scene heading|Scene start|Scene end|Actions and dialogue):',line,re.I): forms[line.split(':')[0]].add(ep)
            pair=PAIR.fullmatch(line.strip())
            if pair:
                for t in pair.groups(): precision[len(t.split('.')[1]) if '.' in t else 0]+=1
            for m in re.finditer(r'\b'+re.escape(a.name_prefix)+r'[a-zA-Z]+\b',line):
                names.append({'episode':ep,'source_file':path.name,'source_line':no,'form':m[0],'text':line})
    a.out.mkdir(parents=True,exist_ok=True)
    write_jsonl(a.out/'name_surface_occurrences.jsonl',names)
    (a.out/'format_inventory.json').write_text(json.dumps({'scene_field_forms':{k:sorted(v) for k,v in forms.items()},
      'event_timestamp_decimal_digits':dict(precision),'name_prefix':a.name_prefix,
      'name_forms':{s:sorted({x['episode'] for x in names if x['form']==s}) for s in sorted({x['form'] for x in names})}},indent=2)+'\n')
    print(f'Inventoried {len(names)} source-located name occurrences; no alias decisions made.')
if __name__=='__main__':main()
