"""Four-kind, reusable offline pools; sampling is separate from pool construction."""
from collections import Counter
import itertools
import json
import re
from pathlib import Path

from filelock import FileLock
import torch

from .config import fingerprint, resolve, file_hash
from .diagnosis_noise import canonical, _rng, _core, _spans, WORD, STOP, spelling_candidates
from .io import write_json

GENERATOR_VERSION = 'diagnosis_four_kind_v6_topk_same_initial'
KINDS = ('detail_omission', 'detail_substitution', 'label_confusion', 'spelling_error')
SAMPLING = 'training seed + record_id + kind + epoch cycle; shuffled round robin; zero-based epoch'


def ranking_loss(positive_nll, negative_nll, margin):
    if negative_nll.ndim != 1 or negative_nll.numel() == 0:
        raise ValueError('Expected a nonempty vector of diagnosis negative NLLs')
    return torch.relu(margin + positive_nll - negative_nll).mean()


def _display(text):
    text = re.sub(r'\s+\(disorder\)$', '', text).strip()
    m = re.fullmatch(r'(.+) - (acute|chronic|bacterial|viral|primary|secondary)', text, re.I)
    if m:
        return m[2].capitalize() + ' ' + m[1][0].lower() + m[1][1:]
    m = re.fullmatch(r'(.+) (sprain|fracture|dislocation)', text, re.I)
    if m:
        return m[2].capitalize() + ' of ' + m[1].lower()
    return text


def _term_key(text):
    return re.sub(r'[^a-z0-9]+', '', canonical(_display(text)))


def _focus(text, rules):
    core = re.split(r'\b(?:of|with|without|due to)\b', _display(text), maxsplit=1, flags=re.I)[0]
    for phrase, replacement in rules.get('focus_phrases', {}).items():
        core = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, core, flags=re.I)
    qualifiers = {canonical(v) for vs in rules['qualifier_groups'].values() for v in vs}
    words = [m.group().casefold() for m in WORD.finditer(core)]
    useful = [w for w in words if w not in STOP | qualifiers | set(rules['lexical_generic_words'])]
    focus = max(useful or words or [''], key=len)
    return rules.get('focus_aliases', {}).get(focus, focus)


def edit_similarity(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return 1 - previous[-1] / max(1, len(a), len(b))


def omission_candidates(gold, rules):
    core, suffix = _core(gold)
    spans = [(m.start(), m.end(), 'qualifier') for vs in rules['qualifier_groups'].values() for m in _spans(core, vs)]
    pattern = r'\b(?:' + '|'.join(map(re.escape, rules['numbered_qualifiers'])) + r')\s+(?:[0-9]+[A-Za-z]?|[IVX]+)\b'
    spans += [(m.start(), m.end(), 'numbered_qualifier') for m in re.finditer(pattern, core, re.I)]
    # Negation stays inside the entire clause; never delete "no" alone.
    spans += [(m.start(), m.end(), 'detail_clause') for m in re.finditer(r'\s+(?:of|with|without|due to)\s+.+$', core, re.I)]
    spans = sorted(set(spans))
    for count in range(1, rules['max_omission_components'] + 1):
        for selected in itertools.combinations(spans, count):
            if any(a[1] > b[0] for a, b in zip(selected, selected[1:])):
                continue
            reduced = core
            for start, end, _ in reversed(selected):
                reduced = reduced[:start] + reduced[end:]
            reduced = ' '.join(reduced.split()).strip(' -')
            if not reduced or _focus(reduced, rules) != _focus(gold, rules):
                continue
            yield {'answer': reduced[0].upper() + reduced[1:] + suffix, 'kind': 'detail_omission',
                   'rule': 'delete_complete_components', 'component_count': count,
                   'changes': [{'start': a, 'end': b, 'before': core[a:b], 'after': '', 'axis': axis} for a, b, axis in selected]}


def body_candidates(gold, resource, rules):
    core, suffix = _core(gold)
    # No attempt to move a disease with a fixed organ (e.g. concussion) to an unrelated one.
    family = next((f for f in resource['body_groups'] if re.search(r'\b' + re.escape(f) + r'\b', core, re.I)), None)
    if family is None:
        return
    group = resource['body_groups'][family]
    if family == 'otitis':
        m = re.search(r'\botitis\s+(media|externa)\b', core, re.I)
        if m:
            original = 'middle ear' if m[1].casefold() == 'media' else 'external ear'
            for item in group:
                if item['value'] != original:
                    replacement = 'media' if item['value'] == 'middle ear' else 'externa'
                    yield {'answer': core[:m.start(1)] + replacement + core[m.end(1):] + suffix,
                           'kind': 'detail_substitution', 'rule': 'shared_compatible_body_pool',
                           'family': family, 'before_site': original, 'after_site': item['value'], 'evidence': item['evidence']}
        return
    m = re.search(r'\bof\s+(.+?)(?=\s+(?:with|without|due to)\b|$)', core, re.I)
    if not m:
        return
    original = rules['site_aliases'].get(canonical(m[1]), canonical(m[1]))
    if original not in {c['value'] for c in group}:
        return
    for item in group:
        if item['value'] != original:
            yield {'answer': core[:m.start(1)] + item['value'] + core[m.end(1):] + suffix,
                   'kind': 'detail_substitution', 'rule': 'shared_compatible_body_pool',
                   'family': family, 'before_site': original, 'after_site': item['value'], 'evidence': item['evidence']}


def _focus_span(text, focus, rules):
    """Find the original disease word/phrase, leaving qualifiers and sites intact."""
    for phrase, replacement in rules.get('focus_phrases', {}).items():
        if replacement == focus:
            match = re.search(r'\b' + re.escape(phrase) + r'\b', text, re.I)
            if match:
                return match.span()
    for match in WORD.finditer(text):
        word = match.group().casefold()
        if rules.get('focus_aliases', {}).get(word, word) == focus:
            return match.span()
    return None


def _word_roots(word):
    roots = {word}
    for suffix in ('s', 'es', 'd', 'ed', 'ing', 'ion', 'ation', 'itis', 'osis', 'al', 'ic', 'ive'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            roots.add(word[:-len(suffix)])
    return roots | {r[:-1] for r in roots if r.endswith('e') and len(r) > 4}


def lexical_candidates(gold, positive_codes, resource, rules, minimum):
    focus = _focus(gold, rules)
    span = _focus_span(gold, focus, rules)
    if not focus or span is None:
        return
    # Match literal letters, including phrase-based diagnoses, rather than a semantic alias.
    literal = gold[span[0]:span[1]]
    word = max(WORD.finditer(literal), key=lambda m: len(m.group()))
    span = span[0] + word.start(), span[0] + word.end()
    focus = gold[span[0]:span[1]].casefold()
    key = _term_key(gold)
    equivalent_ids = {c['id'] for c in resource['conditions']
                      if key in {_term_key(t) for t in [c['name'], *c['aliases']]}
                      or positive_codes & {tuple(code) for code in c['codes']}}
    # Generic omissions remain category 1. Do not relabel their synonyms as a new disease.
    family_keys = {key} | {_term_key(c['answer']) for c in omission_candidates(gold, rules)}
    family_ids = equivalent_ids | {c['id'] for c in resource['conditions']
        if family_keys & {_term_key(t) for t in [c['name'], *c['aliases']]}}
    # A shared coarse code is ambiguous, so conservatively omit it rather than assert a new meaning.
    family_codes = {tuple(code) for c in resource['conditions'] if c['id'] in family_ids for code in c['codes']}
    known_focuses = {_focus(t, rules) for c in resource['conditions'] if c['id'] in equivalent_ids
                     for t in [c['name'], *c['aliases']]}
    known_focuses.add(focus)
    primary_cores = {_focus(c['name'], rules) for c in resource['conditions']}
    known_roots = set().union(*(_word_roots(w) for w in known_focuses))
    for c in resource['conditions']:
        aliases = {_term_key(t) for t in [c['name'], *c['aliases']]}
        if (c['id'] in family_ids or family_codes & {tuple(code) for code in c['codes']}
                or key in aliases):
            continue
        for source_term in dict.fromkeys([c['name'], *c['aliases']]):
            other_focus = _focus(source_term, rules)
            if (not other_focus or other_focus not in primary_cores or other_focus in known_focuses
                    or other_focus[0] != focus[0] or _word_roots(other_focus) & known_roots):
                continue
            score = edit_similarity(focus, other_focus)
            if score < minimum:
                continue
            original = gold[span[0]:span[1]]
            replacement = other_focus.upper() if original.isupper() else (
                other_focus.capitalize() if original[0].isupper() else other_focus)
            answer = gold[:span[0]] + replacement + gold[span[1]:]
            if _term_key(answer) in family_keys:
                continue
            yield {'answer': answer, 'kind': 'label_confusion',
                   'rule': 'same_initial_attested_core_substitution', 'similarity': score,
                   'answer_similarity': edit_similarity(canonical(_core(gold)[0]), canonical(_core(answer)[0])),
                   'focus': focus, 'negative_focus': other_focus,
                   'source': c, 'source_term': source_term, 'generated': True,
                   'changes': [{'start': span[0], 'end': span[1], 'before': original, 'after': replacement}],
                   'meaning_check': 'core word from a different source concept; aliases/codes excluded; '
                                    'generated combination is not an attested or clinician-reviewed diagnosis'}


def _checked_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('fingerprint') != fingerprint({k: v for k, v in value.items() if k != 'fingerprint'}):
        raise ValueError(f'Candidate resource/pool fingerprint mismatch: {path}')
    return value


def _generate_pool(identity, resource, rules):
    options = identity['generation']
    vocabulary = {m.group().casefold() for c in resource['conditions'] for m in WORD.finditer(c['name'])}
    vocabulary.update(m.group().casefold() for row in identity['positives'] for m in WORD.finditer(row['gold']))
    rows = []
    for positive in identity['positives']:
        gold = positive['gold']
        raw = list(omission_candidates(gold, rules)) + list(body_candidates(gold, resource, rules))
        raw.extend(lexical_candidates(gold, {tuple(c) for c in positive['codes']}, resource, rules,
                                      minimum=0.0))
        raw.extend(spelling_candidates(gold, vocabulary, options['pool_seed'], rules))
        pools, seen = {k: [] for k in KINDS}, set()
        # Deterministic ordering of equivalent source labels before global deduplication.
        raw.sort(key=lambda c: (KINDS.index(c['kind']), -c.get('similarity', 0),
                               -c.get('answer_similarity', 0), len(c['answer']), c['answer']))
        for c in raw:
            key = _term_key(c['answer'])
            if key and key != _term_key(gold) and key not in seen:
                pools[c['kind']].append(c)
                seen.add(key)
        available = {k: len(v) for k, v in pools.items()}
        for kind, pool in pools.items():
            if kind != 'label_confusion':
                _rng(options['pool_seed'], gold, kind).shuffle(pool)
            if kind == 'detail_omission':
                pool.sort(key=lambda c: c['component_count'])  # single omissions first
            if kind == 'label_confusion':
                # Prefer different core words to five anatomical variants of the same neighbour.
                focuses = set()
                diverse = []
                for candidate in pool:
                    if candidate['negative_focus'] not in focuses:
                        candidate['above_preferred_similarity'] = candidate['similarity'] >= options['lexical_preferred_similarity']
                        diverse.append(candidate)
                        focuses.add(candidate['negative_focus'])
                pool = diverse
                if len(pool) < options['lexical_candidates_per_diagnosis']:
                    raise ValueError(f'Only {len(pool)} distinct same-initial lexical candidates for {gold!r}; '
                                     f"need {options['lexical_candidates_per_diagnosis']}. Expand the terminology resource.")
            if kind == 'spelling_error':
                selected = []
                for count in (1, 1, 1, 2, 3):
                    candidate = next((c for c in pool if c['edit_count'] == count and c not in selected), None)
                    if candidate:
                        selected.append(candidate)
                pool = selected + [c for c in pool if c not in selected]
            count = options['lexical_candidates_per_diagnosis'] if kind == 'label_confusion' else options['candidates_per_kind']
            pools[kind] = pool[:count]
        if not any(pools.values()):
            raise ValueError(f'No diagnosis negatives for {gold!r}')
        rows.append({'gold': gold, 'pools': pools, 'available_before_cap': available})
    payload = {'identity': identity, 'by_diagnosis': rows,
               'metadata': {'version': GENERATOR_VERSION, 'kinds': list(KINDS),
                            'shared_body_pool_sizes': {k: len(v) for k, v in resource['body_groups'].items()},
                            'shared_body_pools': resource['body_groups'],
                            'positive_diagnoses': len(rows), 'test_overlap_policy': 'allowed; no test files read',
                            'lexical_quality': {r['gold']: {
                                'count': len(r['pools']['label_confusion']),
                                'preferred_similarity': options['lexical_preferred_similarity'],
                                'above_preferred_count': sum(c['above_preferred_similarity'] for c in r['pools']['label_confusion']),
                                'minimum_similarity': min(c['similarity'] for c in r['pools']['label_confusion'])}
                                for r in rows},
                            'limits': 'Heuristic terminology and component checks; lexical closeness does not establish empirical model hardness. '
                                      'Lexical negatives use the top K distinct same-initial cores from source names/aliases; '
                                      'the preferred similarity is a quality marker, not a cutoff. '
                                      'The whole generated combination is not necessarily an attested diagnosis. '
                                      'Explicit aliases/codes are excluded; source terminology is not an exhaustive synonym oracle.'}}
    return dict(payload, fingerprint=fingerprint(payload))


def load_or_build_pool(records, cfg):
    options = cfg['diagnosis_contrastive']
    resource_path, rules_path = resolve(options['resources_file']), resolve(options['rules_file'])
    resource = _checked_json(resource_path)
    rules = json.loads(rules_path.read_text(encoding='utf-8'))
    by_label = {}
    for r in records:
        p = r.get('provenance', {})
        code = p.get('source_condition_system'), p.get('source_condition_code')
        codes = by_label.setdefault(r['fields']['diagnosis'], set())
        if all(code):
            codes.add(code)
    identity = {'version': GENERATOR_VERSION, 'resources_sha256': file_hash(resource_path),
                'generator_code_sha256': fingerprint([file_hash(Path(__file__)), file_hash(Path(__file__).with_name('diagnosis_noise.py'))]),
                'rules_sha256': file_hash(rules_path),
                'generation': {k: options[k] for k in ('pool_seed', 'candidates_per_kind', 'lexical_preferred_similarity', 'lexical_candidates_per_diagnosis')},
                'positives': [{'gold': gold, 'codes': [list(c) for c in sorted(codes)]} for gold, codes in sorted(by_label.items())]}
    directory = resolve(options['pool_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (fingerprint(identity) + '.json')
    # A lock prevents duplicate construction by simultaneous runs or multiple GPU ranks.
    with FileLock(str(directory / 'build.lock')):
        if path.exists():
            pool = _checked_json(path)
            if pool['identity'] != identity:
                raise ValueError('Cached candidate pool identity mismatch')
            event = 'hit'
        else:
            pool = _generate_pool(identity, resource, rules)
            write_json(path, pool)
            event = 'built'
    print(json.dumps({'diagnosis_pool_cache': event, 'pool_file': str(path), 'fingerprint': pool['fingerprint']}), flush=True)
    # These references contain no cache-hit flag/timestamp, so run/resume identities stay stable.
    return pool, {'directory': options['pool_directory'], 'file': path.name, 'fingerprint': pool['fingerprint']}


def select_negatives(row, *, epoch=0):
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Negative sampling epoch must be a nonnegative integer')
    chosen = []
    for kind in KINDS:
        pool = row['pools'][kind]
        if pool:
            cycle, offset = divmod(epoch, len(pool))
            order = list(range(len(pool)))
            _rng(row['seed'], row['record_id'], kind, cycle).shuffle(order)
            chosen.append(pool[order[offset]])
    return chosen


def build_bank(records, cfg):
    if not records or len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Negative preparation needs nonempty unique training records')
    if any(r.get('split', 'train') not in ('train', 'dev') for r in records):
        raise ValueError('Negative pools may only bind actual training records')
    shared, reference = load_or_build_pool(records, cfg)
    by_gold = {r['gold']: r for r in shared['by_diagnosis']}
    entries = []
    for record in records:
        gold = record['fields']['diagnosis']
        row = {'record_id': record['record_id'], 'gold': gold, 'seed': cfg['seed'], 'pools': by_gold[gold]['pools']}
        row['negatives'] = select_negatives(row, epoch=0)
        entries.append(row)
    coverage = {k: sum(bool(r['pools'][k]) for r in entries) for k in KINDS}
    metadata = {'generator_version': GENERATOR_VERSION, 'records': len(records), 'positive_labels': len(by_gold),
                'pool_seed': cfg['diagnosis_contrastive']['pool_seed'], 'sampling_seed': cfg['seed'],
                'shared_pool': reference, 'kinds': list(KINDS), 'max_negatives_per_record': 4,
                'candidates_per_kind': cfg['diagnosis_contrastive']['candidates_per_kind'],
                'lexical_candidates_per_diagnosis': cfg['diagnosis_contrastive']['lexical_candidates_per_diagnosis'],
                'lexical_quality': shared['metadata']['lexical_quality'],
                'negative_answers': sum(len(r['negatives']) for r in entries),
                'negative_count_histogram': dict(Counter(str(len(r['negatives'])) for r in entries)),
                'selected_kinds_by_record': coverage, 'coverage_fraction': {k: n/len(records) for k,n in coverage.items()},
                'all_four_kinds_covered': all(coverage.values()),
                'records_with_all_four_kinds': sum(all(r['pools'].values()) for r in entries),
                'missing_kinds_by_positive': {g: [k for k in KINDS if not r['pools'][k]] for g,r in by_gold.items()},
                'pool_sizes_by_positive': {g: {k: len(v) for k,v in r['pools'].items()} for g,r in by_gold.items()},
                'available_before_cap': {g: r['available_before_cap'] for g,r in by_gold.items()},
                'shared_body_pool_sizes': shared['metadata']['shared_body_pool_sizes'],
                'test_overlap_policy': 'allowed; no test answers or predictions loaded',
                'selection': SAMPLING, 'preview_epoch': 1, 'limits': shared['metadata']['limits']}
    payload = {'metadata': metadata, 'entries': entries}
    return dict(payload, fingerprint=fingerprint(payload))


def persist_bank(bank, output, *, resume=False):
    out = resolve(output)
    path = out / 'diagnosis_negatives.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != bank:
            raise ValueError('Saved diagnosis negative bank differs from current records/pool/sampling seed')
    elif resume:
        raise ValueError('Resume requires the original diagnosis_negatives.json')
    else:
        write_json(path, bank)
    report = dict(bank['metadata'], fingerprint=bank['fingerprint'], status='PASS')
    audit = out / 'diagnosis_negative_audit.json'
    if audit.exists() and json.loads(audit.read_text(encoding='utf-8')) != report:
        raise ValueError('Saved diagnosis negative audit differs from current bank')
    if not audit.exists():
        write_json(audit, report)
    examples = [{'record_id': r['record_id'], 'gold': r['gold'], 'pools': r['pools'],
                 'selections': {f'epoch_{e+1}': select_negatives(r, epoch=e) for e in range(3)}} for r in bank['entries'][:20]]
    write_json(out / 'diagnosis_negative_examples.json', examples)


def bank_index(bank):
    return {r['record_id']: r for r in bank['entries']}
