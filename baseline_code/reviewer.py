"""Swappable review adapter extension point. Not called by candidates.py.

A future provider implements review(candidate, episodes) -> dict. Only source IDs
may be returned as evidence; this stage never permits model-authored timestamps.
OfflineProvider is deliberately not a fake judge: it abstains on semantic quality.
"""
import importlib
import json
from pathlib import Path
from typing import Protocol

class ReviewProvider(Protocol):
    def review(self, candidate: dict, episodes: dict) -> dict: ...

class OfflineProvider:
    def __init__(self, **options): pass
    def review(self,candidate,episodes):
        return {'candidate_id':candidate['id'],'status':'not_reviewed',
                'hook':None,'cliffhanger':None,'standalone':None,
                'reason':'Offline mode does not claim semantic approval.', 'evidence_event_ids':[]}

def load_provider(config_path=Path(__file__).with_name('review_config.json')) -> ReviewProvider:
    config=json.loads(Path(config_path).read_text())
    name=config['provider']
    if name=='offline': return OfflineProvider(**config.get('options',{}))
    # Trusted local configuration only: importing an adapter executes its Python code.
    module,cls=name.split(':',1)
    return getattr(importlib.import_module(module),cls)(**config.get('options',{}))
