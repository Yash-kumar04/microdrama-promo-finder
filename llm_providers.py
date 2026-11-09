"""HTTP adapters, offline fixture provider, content-addressed cache and usage audit."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

VERSION = 'kuku-review-v2.2'

class ProviderError(RuntimeError):
    pass

def load_config(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('Use JSON-compatible YAML or install PyYAML for general YAML syntax') from exc
        cfg = yaml.safe_load(text)
    if cfg.get('provider') not in ('mock', 'gemini', 'openai_compatible'):
        raise ValueError('provider must be mock, gemini, or openai_compatible')
    if cfg.get('max_attempts', 0) < 1 or cfg.get('max_promos_per_episode', 0) < 1:
        raise ValueError('max_attempts and max_promos_per_episode must be positive')
    if any(cfg.get(k, 0) < 0 for k in ('min_request_interval_seconds', 'retry_base_seconds', 'max_retry_sleep_seconds')):
        raise ValueError('Sleep settings must be nonnegative')
    if cfg.get('timeout_seconds', 0) <= 0 or cfg.get('max_prompt_characters', 0) <= 0:
        raise ValueError('Timeout and prompt budget must be positive')
    return cfg

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)

def parse_response(text):
    # Permit a single surrounding Markdown fence, but never scrape prose for braces.
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    value = json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
    if not isinstance(value, dict):
        raise ValueError('Response must be one JSON object')
    return value

def mock_response(stage, payload):
    """Transport/schema fixtures, explicitly NOT editorial judgments."""
    if stage == 'A':
        rows, accepted = [], 0
        events = payload['events']
        for c in payload['candidates']:
            early = c['hook_eligible_event_ids']
            tail = c['ending_eligible_event_ids']
            ok = bool(early and tail and accepted < payload['max_accept'])
            if ok:
                accepted += 1
                h, t = early[0], tail[-1]
            else:
                h = t = ''
            rows.append({'candidate_id': c['candidate_id'], 'decision': 'accept' if ok else 'reject',
                         'segments': [], 'hook_event_id': h, 'hook_text': events[h]['text'] if h else '',
                         'hook_why': '[MOCK] Schema fixture; no semantic hook judgment.',
                         'cliffhanger_event_id': t, 'cliffhanger_text': events[t]['text'] if t else '',
                         'cliffhanger_why': '[MOCK] Schema fixture; no cliffhanger judgment.',
                         'why_it_works': '[MOCK] Demonstrates ID selection and validation only.',
                         'scores': {'hook': 3, 'cliffhanger': 3, 'standalone': 3, 'intensity': 3},
                         'resolved_story_goal': False, 'rhetorical_question': False,
                         'reject_reason': '' if ok else '[MOCK] Not selected by the deterministic fixture.'})
        return {'reviews': rows}
    if stage == 'B':
        # Deterministic fixture includes both pass and fail code paths; not a prediction.
        fail = int(hashlib.sha256(payload['clip_text'].encode()).hexdigest()[:2], 16) % 5 == 0
        return {'confusing': ['[MOCK] Simulated confusion'] if fail else [],
                'missing_context': ['[MOCK] Simulated missing context'] if fail else [],
                'would_keep_watching': 'N' if fail else 'Y',
                'what_happens_next': '[MOCK] No actual cold-viewer prediction was made.'}
    ordered = sorted(payload['promos'], key=lambda x: (-sum(x['scores'].values()), x['promo_id']))
    return {'ranked': [{'promo_id': x['promo_id'], 'reason': '[MOCK] Stable fixture ordering, not a posting recommendation.'} for x in ordered],
            'top_3': [x['promo_id'] for x in ordered[:3]]}

class Client:
    def __init__(self, cfg, output_dir, transport=None, sleep=time.sleep, clock=time.monotonic):
        self.cfg, self.out = cfg, Path(output_dir)
        self.cache = Path(cfg['cache_dir'])
        self.cache.mkdir(parents=True, exist_ok=True)
        self.raw = self.out/'raw'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.transport = transport or self._http
        self.sleep, self.clock, self.last_request = sleep, clock, None
        self.calls, self.disabled = [], None

    @staticmethod
    def _http(url, headers, body, timeout):
        request = urllib.request.Request(url, data=canonical(body).encode(), headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def request_parts(self, stage, system, payload):
        cfg = self.cfg
        provider, base = cfg['provider'], cfg['base_url'].rstrip('/')
        key = os.environ.get(cfg.get('api_key_env', ''), '')
        if provider == 'gemini' and not key:
            raise ProviderError('Missing API key environment variable; no mock fallback')
        if provider == 'openai_compatible' and cfg.get('api_key_env') and not key:
            raise ProviderError('Missing API key environment variable; set api_key_env empty for unauthenticated local endpoints')
        headers = {'Content-Type': 'application/json'}
        if provider == 'gemini':
            headers['x-goog-api-key'] = key
            url = base+'/models/'+urllib.parse.quote(cfg['model'], safe='')+':generateContent'
            generation = {'temperature': cfg['temperature'], 'maxOutputTokens': cfg['max_output_tokens'][stage],
                          'responseMimeType': 'application/json'}
            generation.update(cfg.get('provider_options', {}))
            body = {'systemInstruction': {'parts': [{'text': system}]},
                    'contents': [{'role': 'user', 'parts': [{'text': canonical(payload)}]}],
                    'generationConfig': generation}
        else:
            if key:
                headers['Authorization'] = 'Bearer '+key
            url = base+'/chat/completions'
            body = {'model': cfg['model'], 'temperature': cfg['temperature'],
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical(payload)}],
                    cfg.get('openai_token_parameter', 'max_tokens'): cfg['max_output_tokens'][stage]}
            if cfg.get('openai_json_mode', True):
                body['response_format'] = {'type': 'json_object'}
            body.update(cfg.get('provider_options', {}))
        return url, headers, body

    def decode(self, response):
        if self.cfg['provider'] == 'gemini':
            choices = response.get('candidates') or []
            if not choices:
                raise ValueError('Provider returned no candidate (possibly blocked)')
            if choices[0].get('finishReason') not in (None, 'STOP'):
                raise ValueError('Provider response incomplete or blocked')
            text = ''.join(p.get('text', '') for p in choices[0].get('content', {}).get('parts', []) if not p.get('thought'))
            u = response.get('usageMetadata', {})
            inp = u.get('promptTokenCount')
            out = (u.get('candidatesTokenCount', 0)+u.get('thoughtsTokenCount', 0)) if u else None
            return text, inp, out
        choice = response['choices'][0]
        if choice.get('finish_reason') not in (None, 'stop'):
            raise ValueError('Provider response incomplete or blocked')
        u = response.get('usage', {})
        return choice['message']['content'], u.get('prompt_tokens'), u.get('completion_tokens')

    def call(self, stage, label, system, payload, validate=None):
        cfg = self.cfg
        request = {'version': VERSION, 'provider': cfg['provider'], 'model': cfg['model'],
                   'base_url': cfg['base_url'], 'temperature': cfg['temperature'],
                   'max_output_tokens': cfg['max_output_tokens'][stage],
                   'provider_options': cfg.get('provider_options', {}),
                   'openai_json_mode': cfg.get('openai_json_mode'), 'openai_token_parameter': cfg.get('openai_token_parameter'),
                   'stage': stage, 'system': system, 'payload': payload}
        key = hashlib.sha256(canonical(request).encode()).hexdigest()
        cachefile = self.cache/(key+'.json')
        rawfile = self.raw/(stage+'_'+label+'_'+key[:10]+'.json')
        prompt_chars = len(system)+len(canonical(payload))
        log = {'stage': stage, 'label': label, 'cache_key': key, 'provider': cfg['provider'],
               'model': cfg['model'], 'prompt_characters': prompt_chars, 'cached': False, 'attempts': [],
               'billable_input_tokens_estimate': 0, 'billable_output_tokens_estimate': 0, 'estimated_cost_usd': 0.0}
        self.calls.append(log)
        record = {'request': request, 'attempts': []}
        if prompt_chars > cfg['max_prompt_characters']:
            log['error'] = 'prompt_budget_exceeded'
            atomic_json(rawfile, record)
            raise ProviderError(log['error'])
        if cachefile.exists():
            try:
                cached = json.loads(cachefile.read_text())
                if cached['cache_key'] == key:
                    result = parse_response(cached['response_text'])
                    if validate is not None:
                        validate(result)
                    log.update(cached=True, status='ok', input_tokens=cached['input_tokens'], output_tokens=cached['output_tokens'],
                               usage_basis=cached['usage_basis'])
                    atomic_json(rawfile, dict(record, cached_response=cached))
                    return result
            except (ValueError, KeyError, TypeError):
                log['cache_warning'] = 'corrupt_cache_ignored'
        if self.disabled:
            log['error'] = self.disabled
            atomic_json(rawfile, record)
            raise ProviderError(self.disabled)
        last_error = 'unknown'
        for attempt in range(cfg['max_attempts']):
            entry = {'attempt': attempt+1}
            log['attempts'].append(entry)
            try:
                if cfg['provider'] == 'mock':
                    response_text = canonical(mock_response(stage, payload))
                    response, inp, out = {'mock': True, 'text': response_text}, 0, 0
                else:
                    url, headers, body = self.request_parts(stage, system, payload)
                    if self.last_request is not None:
                        delay = max(0, cfg['min_request_interval_seconds']-(self.clock()-self.last_request))
                        if delay:
                            self.sleep(delay)
                    self.last_request = self.clock()
                    response = self.transport(url, headers, body, cfg['timeout_seconds'])
                    record['attempts'].append({'attempt': attempt+1, 'provider_response': response})
                    if not isinstance(response, dict):
                        raise ValueError('provider_response_not_object')
                    # Account for tokens even if response is truncated, refused or invalid JSON.
                    if cfg['provider'] == 'gemini':
                        u = response.get('usageMetadata', {})
                        inp = u.get('promptTokenCount')
                        out = u.get('candidatesTokenCount', 0)+u.get('thoughtsTokenCount', 0) if u else None
                    else:
                        u = response.get('usage', {})
                        inp, out = u.get('prompt_tokens'), u.get('completion_tokens')
                    basis = 'provider' if inp is not None and out is not None else 'estimated_chars_div_4'
                    inp = inp if inp is not None else math.ceil(prompt_chars/4)
                    out = out if out is not None else math.ceil(len(canonical(response))/4)
                    log['billable_input_tokens_estimate'] += inp
                    log['billable_output_tokens_estimate'] += out
                    rates = cfg['pricing']
                    log['estimated_cost_usd'] += (inp*rates['input_usd_per_million']+out*rates['output_usd_per_million'])/1e6
                    response_text, _, _ = self.decode(response)
                if cfg['provider'] == 'mock':
                    basis = 'offline_zero_tokens'
                    record['attempts'].append({'attempt': attempt+1, 'provider_response': response})
                result = parse_response(response_text)
                if validate is not None:
                    validate(result)
                cached = {'cache_key': key, 'response_text': response_text, 'input_tokens': inp,
                          'output_tokens': out, 'usage_basis': basis}
                atomic_json(cachefile, cached)
                atomic_json(rawfile, record)
                log.update(status='ok', input_tokens=inp, output_tokens=out, usage_basis=basis)
                return result
            except urllib.error.HTTPError as exc:
                last_error = 'http_'+str(exc.code)  # Never log keys, request headers or server echo text.
                entry['error'] = last_error
                retry = exc.code == 429 or exc.code in (500, 502, 503, 504)
                if exc.code in (401, 403):
                    self.disabled = last_error
                delay = cfg['retry_base_seconds']*(2**attempt)
                retry_after = exc.headers.get('Retry-After') if exc.headers else None
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            date = parsedate_to_datetime(retry_after)
                            delay = max(delay, (date-datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError):
                            pass
                if delay > cfg['max_retry_sleep_seconds']:
                    # Do not hammer a quota endpoint earlier than Retry-After.
                    self.disabled = 'retry_after_exceeds_sleep_budget'
                    retry = False
            except ProviderError:
                last_error = 'missing_or_invalid_provider_configuration'
                entry['error'] = last_error
                self.disabled, retry, delay = last_error, False, 0
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                last_error = 'invalid_provider_response_'+type(exc).__name__
                entry['error'] = last_error
                retry, delay = True, cfg['retry_base_seconds']*(2**attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                last_error = 'network_error'
                entry['error'] = last_error
                retry, delay = True, cfg['retry_base_seconds']*(2**attempt)
            atomic_json(rawfile, record)
            if not retry or attempt+1 == cfg['max_attempts']:
                break
            entry['sleep_seconds'] = min(delay, cfg['max_retry_sleep_seconds'])
            self.sleep(entry['sleep_seconds'])
        log.update(status='failed', error=last_error)
        raise ProviderError(last_error)
