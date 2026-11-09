"""Post-process promos_ranked.json: drop overlapping promos and force the top 3 into different episodes.
Usage: python postprocess_diverse.py path/to/outputs_llm_colab
Reads promos_ranked.json (never modified) and writes promos_final_diverse.csv and promos_dropped_as_overlap.csv."""
import csv, json, sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs_llm_colab')
promos = sorted(json.loads((folder / 'promos_ranked.json').read_text())['promos'], key=lambda p: p['rank'])

def overlaps(a, b):
    return any(x['episode'] == y['episode'] and x['start_sec'] < y['end_sec'] and y['start_sec'] < x['end_sec']
               for x in a['segments'] for y in b['segments'])

def spoken(s):
    return s.strip().startswith(('"', '\u201c'))

kept, dropped = [], []
for p in promos:
    clash = next((k for k in kept if overlaps(p, k)), None)
    if clash:
        dropped.append((p, clash['rank']))
    else:
        kept.append(p)

top, used = [], set()
for p in kept:
    ep = p['segments'][0]['episode']
    if ep not in used and len(top) < 3:
        top.append(p)
        used.add(ep)
final = top + [p for p in kept if p not in top]

rows = [{'new_rank': i, 'original_rank': p['rank'], 'post_first': i <= 3, 'segments': p['segments_formatted'],
         'total_seconds': p['total_seconds'], 'hook': p['hook'], 'cliffhanger': p['cliffhanger'],
         'why_it_works': p['why_it_works'],
         'needs_label_check': not (spoken(p['hook']) and spoken(p['cliffhanger']))}
        for i, p in enumerate(final, 1)]

with open(folder / 'promos_final_diverse.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['new_rank', 'original_rank', 'post_first', 'segments', 'total_seconds', 'hook', 'cliffhanger', 'why_it_works', 'needs_label_check']); w.writeheader(); w.writerows(rows)
with open(folder / 'promos_dropped_as_overlap.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['dropped_original_rank', 'overlaps_with_original_rank', 'segments'])
    for p, r in dropped:
        w.writerow([p['rank'], r, p['segments_formatted']])

if not rows:
    print('No promos survived; wrote headers-only CSVs.')

print(len(promos), 'ranked ->', len(final), 'kept,', len(dropped), 'dropped as overlaps')
for r in rows[:3]:
    print(r['new_rank'], r['segments'], f"(was #{r['original_rank']})")
print('Needs label check:', sum(r['needs_label_check'] for r in rows), 'of', len(rows))
