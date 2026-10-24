"""Deterministic text helpers and multi-position nonword corruption."""
import hashlib
import random
import re
import string
import unicodedata

WORD = re.compile(r"[A-Za-z]+")
STOP = {"of", "the", "with", "a", "an", "in", "and", "or", "to", "by", "due", "no"}

def canonical(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _rng(*parts):
    return random.Random(int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest(), 16))


def _core(label):
    tag = re.search(r'\s+\([^()]+\)$', label)
    return (label[:tag.start()], label[tag.start():]) if tag else (label, '')


def _spans(text, values):
    if not values:
        return []
    pattern = r'(?<![A-Za-z])(?:' + '|'.join(re.escape(v) for v in sorted(values, key=lambda v: (-len(v), v))) + r')(?![A-Za-z])'
    return list(re.finditer(pattern, text, re.I))


def spelling_candidates(gold, vocabulary, seed, rules):
    """Mix 1/2/3 edits across positions/words; keep punctuation, numbers and tags."""
    core, suffix = _core(gold)
    matches = list(WORD.finditer(core))
    qualifiers = {canonical(v) for values in rules['qualifier_groups'].values() for v in values}
    eligible = [i for i, m in enumerate(matches) if len(m.group()) >= 4 and canonical(m.group()) not in STOP | qualifiers]
    if not eligible:
        return
    originals = [m.group() for m in matches]
    length = sum(len(originals[i]) for i in eligible)
    maximum = 3 if length >= 10 else 2 if length >= 6 else 1
    rng = _rng(seed, gold, 'spelling_pool')
    operations = ('substitute', 'delete', 'insert', 'duplicate', 'transpose', 'reverse_segment')
    for attempt in range(180):
        count = rng.choice([1] * 6 + ([2] * 3 if maximum >= 2 else []) + ([3] if maximum >= 3 else []))
        words, changes = originals[:], []
        for _ in range(count):
            index = rng.choice(eligible)
            word = words[index]
            op = rng.choice(operations) if len(word) >= 4 else 'insert'
            pos = rng.randrange(1, len(word))
            if op == 'substitute':
                char = rng.choice(string.ascii_lowercase.replace(word[pos].lower(), ''))
                changed = word[:pos] + (char.upper() if word[pos].isupper() else char) + word[pos+1:]
            elif op == 'delete':
                changed = word[:pos] + word[pos+1:]
            elif op in ('insert', 'duplicate'):
                char = rng.choice(string.ascii_lowercase) if op == 'insert' else word[pos]
                changed = word[:pos] + char + word[pos:]
            else:
                width = 2 if op == 'transpose' else rng.choice((2, 3))
                pos = min(pos, len(word) - width)
                changed = word[:pos] + word[pos:pos+width][::-1] + word[pos+width:]
            if changed != word:
                changes.append({'operation': op, 'word_index': index, 'position': pos, 'before': word, 'after': changed})
                words[index] = changed
        if len(changes) != count or words == originals:
            continue
        changed_indices = [i for i in eligible if words[i] != originals[i]]
        if any(canonical(words[i]) in vocabulary for i in changed_indices):
            continue
        if any(sum(c['word_index'] == i for c in changes) > max(1, (len(originals[i]) + 3) // 4) for i in changed_indices):
            continue
        pieces, end = [], 0
        for m, word in zip(matches, words):
            pieces.extend((core[end:m.start()], word))
            end = m.end()
        yield {'answer': ''.join(pieces) + core[end:] + suffix, 'kind': 'spelling_error',
               'rule': 'random_multiedit', 'edit_count': count, 'changes': changes}
