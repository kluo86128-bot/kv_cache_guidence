import json
import random
from pathlib import Path
from dataclasses import replace
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .cache import Context, causal_mask, fresh_cache, pairs, slice_layers
from .config import file_hash, fingerprint
from .tasks import grouped_loss, questions_for


class Backend:
    def __init__(self, model, tokenizer, cfg, *, identity=None):
        if model.config.model_type != 'qwen3':
            raise ValueError('v1 is validated for Qwen3 only')
        if any(kind != 'full_attention' for kind in getattr(model.config, 'layer_types', [])):
            raise ValueError('v1 requires full-attention Qwen3 layers; sliding caches are not validated')
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.model.eval()
        self.model.requires_grad_(False)
        self.embedding = model.get_input_embeddings()
        self.device, self.dtype = self.embedding.weight.device, self.embedding.weight.dtype
        self.public_ids = self.ids(cfg['public_text'])
        self.separator_ids = self.ids(cfg['separator'])
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError('Tokenizer needs an EOS token')
        self.eos = int(eos)
        stops = model.generation_config.eos_token_id
        self.stops = set(stops if isinstance(stops, (list, tuple)) else [stops]) | {self.eos}
        self.stops.discard(None)
        self.identity = identity or {'model_type': model.config.model_type, 'model_config': fingerprint(model.config.to_dict()),
                                     'tokenizer': fingerprint(tokenizer.get_vocab())}

    @classmethod
    def load(cls, cfg, expected_tokenizer_sha256):
        import transformers
        if transformers.__version__ not in ('4.57.1', '4.57.6'):
            raise RuntimeError('Use transformers==4.57.1 (kv_priv server) or 4.57.6 (local checks); other Cache/mask APIs are not validated')
        if torch.device(cfg['device']).type == 'cuda':
            torch.cuda.set_device(torch.device(cfg['device']))
        kwargs = {'revision': cfg['model_revision'], 'local_files_only': cfg['local_files_only'], 'trust_remote_code': False}
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name_or_path'], **kwargs)
        actual_sha = __import__('hashlib').sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        # JSON whitespace may differ when reserialized: compare parsed JSON in that case.
        local_tokenizer = Path(cfg['model_name_or_path']) / 'tokenizer.json'
        from .config import ROOT
        expected_path = ROOT / 'tools/tokenizer/Qwen3-4B/tokenizer.json'
        if local_tokenizer.is_file() and file_hash(local_tokenizer) == expected_tokenizer_sha256:
            actual_sha = expected_tokenizer_sha256
        elif expected_path.is_file() and file_hash(expected_path) == expected_tokenizer_sha256:
            actual = json.loads(tokenizer.backend_tokenizer.to_str())
            if actual != json.loads(expected_path.read_text(encoding='utf-8')):
                raise ValueError('Model tokenizer differs from the frozen dataset tokenizer snapshot')
            actual_sha = expected_tokenizer_sha256
        else:
            raise ValueError('Cannot verify tokenizer against the frozen snapshot')
        model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'],
                torch_dtype=getattr(torch, cfg['dtype']), attn_implementation=cfg['attention_implementation'], **kwargs)
        model.to(torch.device(cfg['device']))
        model_path = Path(cfg['model_name_or_path'])
        files = {}
        if model_path.is_dir():
            for p in sorted(model_path.iterdir()):
                if p.suffix in ('.safetensors', '.bin'):
                    files[p.name] = {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
        identity = {'model': cfg['model_name_or_path'], 'requested_revision': cfg['model_revision'],
                    'torch_version': str(torch.__version__), 'transformers_version': transformers.__version__,
                    'resolved_revision': getattr(model.config, '_commit_hash', None),
                    'config_sha256': fingerprint(model.config.to_dict()), 'local_weight_inventory': files,
                    'weight_identity_scope': 'config/revision and file size+mtime inventory, not a full weight-byte checksum',
                    'tokenizer_sha256': actual_sha}
        return cls(model, tokenizer, cfg, identity=identity)

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tensor(self, ids):
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def question_text(self, question):
        if self.cfg['question_format'] == 'plain':
            return f'Question: {question}\nAnswer:\n'
        return self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def question_and_target(self, qa):
        text = self.question_text(qa['question'])
        q = self.ids(text)
        joined = self.ids(text + qa['answer'])
        if joined[:len(q)] != q:
            raise ValueError('Question/answer boundary retokenizes; choose a stable prompt boundary')
        y = joined[len(q):] + [self.eos]
        empty_component = (self.cfg.get('name_mode', 'full') == 'parts'
                           and qa.get('field') in ('name_first', 'name_middle', 'name_last')
                           and qa['answer'] == ''
                           and (qa.get('empty_target') is True or qa['field'] == 'name_middle'))
        if len(y) < 2 and not empty_component:
            raise ValueError('Empty target answer')
        return q, y

    def serialized_ids(self, record, prompt_ids):
        before = record['private_prefix'] + self.cfg['separator']
        left = self.private_ids(record)
        middle = self.separator_ids if prompt_ids else []
        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self.ids(text) != list(prompt_ids):
            raise ValueError('Prompt token IDs do not survive decode/encode')
        whole = before + text + (self.cfg['separator'] if prompt_ids else '') + self.cfg['public_text']
        expected = left + list(prompt_ids) + middle + self.public_ids
        if self.ids(whole) != expected:
            raise ValueError('Serialized hard prompt changes full-context token boundaries')
        return expected

    def initial_ids(self):
        pool = []
        for word in self.cfg['initialization_words']:
            ids = self.ids(' ' + word)
            if len(ids) == 1 and ids[0] not in self.tokenizer.all_special_ids:
                pool.append(ids[0])
        if not pool:
            raise ValueError('Initialization vocabulary has no stable ordinary single tokens')
        rng = random.Random(self.cfg['seed'])
        return rng.choices(pool, k=self.cfg['prompt_length'])

    def forward(self, *, ids=None, embeddings=None, layers=(), position_start=0, position_ids=None, mask=None, logits_slice=None):
        if (ids is None) == (embeddings is None):
            raise ValueError('Specify exactly one of ids/embeddings')
        x = (ids if torch.is_tensor(ids) else self.tensor(ids)) if ids is not None else embeddings
        count = x.shape[1]
        past = layers[0][0].shape[2] if layers else 0
        positions = (torch.arange(position_start, position_start + count, device=self.device)[None]
                     if position_ids is None else position_ids)
        physical = torch.arange(past, past + count, device=self.device)
        if mask is None:
            mask = causal_mask(count, past, device=self.device, dtype=self.dtype)
        inputs = {'input_ids': x} if ids is not None else {'inputs_embeds': x}
        result = self.model.model(**inputs, past_key_values=fresh_cache(layers) if layers else None,
                position_ids=positions, cache_position=physical, attention_mask=mask,
                use_cache=True, return_dict=True)
        logits = None
        if logits_slice is not None:
            hidden = result.last_hidden_state
            if torch.is_tensor(logits_slice):
                hidden = hidden.gather(1, logits_slice[..., None].expand(-1, -1, hidden.shape[-1]))
            else:
                hidden = hidden[:, logits_slice]
            logits = self.model.lm_head(hidden).float()
        return logits, pairs(result.past_key_values)

    def private_ids(self, record):
        cached=record.get('_input_cache')
        if cached is None:
            return self.ids(record['private_prefix']+self.cfg['separator'])
        from .input_cache import input_hash
        if (cached['input_sha256'] != input_hash(record['private_prefix']+self.cfg['separator'])
                or cached['tokenizer_sha256'] != self.identity.get('tokenizer_sha256')):
            raise ValueError('Cached input does not match current text/tokenizer')
        return cached['ids']

    def layout(self, record, *, vectors=None, prompt_ids=None):
        if vectors is not None and prompt_ids is not None:
            raise ValueError('Use soft vectors OR hard prompt IDs')
        left = self.private_ids(record)
        if vectors is not None:
            prompt = vectors.to(device=self.device, dtype=self.dtype)
            if prompt.ndim == 2:
                prompt = prompt[None]
            size = prompt.shape[1]
        else:
            prompt_ids = list(prompt_ids or [])
            self.serialized_ids(record, prompt_ids)
            prompt = self.embedding(self.tensor(prompt_ids))
            size = len(prompt_ids)
        middle = self.separator_ids if size else []
        start = len(left) + size + len(middle)
        chunks = [self.embedding(self.tensor(left)), prompt,
                  self.embedding(self.tensor(middle)), self.embedding(self.tensor(self.public_ids))]
        return torch.cat(chunks, dim=1), start

    def context(self, record, *, vectors=None, prompt_ids=None, full=False, blocked=False):
        embeddings, start = self.layout(record, vectors=vectors, prompt_ids=prompt_ids)
        end = embeddings.shape[1]
        if blocked:
            if full:
                raise ValueError('Blocked and full-context are different conditions')
            # C-only encoding at original absolute positions severs every direct
            # and indirect P/U -> C path without editing the source record.
            _, layers = self.forward(ids=self.public_ids, position_start=start)
        else:
            if self.cfg.get('split_private_prefill', False):
                # The frozen causal private prefix cannot depend on later U.
                # Keep its KV as constants; avoid a long-prefix autograd graph.
                private_ids = self.private_ids(record)
                with torch.no_grad():
                    _, private_layers = self.forward(ids=private_ids)
                _, layers = self.forward(embeddings=embeddings[:, len(private_ids):],
                                         layers=private_layers, position_start=len(private_ids))
            else:
                _, layers = self.forward(embeddings=embeddings)
            if not full:
                layers = slice_layers(layers, start, end)
        context = Context(layers, end, start, len(self.public_ids), full)
        expected = end if full else len(self.public_ids)
        if context.physical_length != expected:
            raise AssertionError('Cache slice includes non-public positions')
        return context

    def _mean_answer_losses(self, logits, labels, *, training=False):
        """Return NLL, or [Focal/CE, NLL] for each answer during training.

        Reduction is by valid token count (including EOS), never by focal weights.
        Evaluation callers use the default NLL regardless of the training config.
        Joint training retains both gradients and applies the field mask later.
        """
        ce = F.cross_entropy(logits.float().transpose(1, 2), labels,
                             reduction='none', ignore_index=-100)
        valid = labels.ne(-100)
        lengths = valid.sum(dim=1)
        nll = ce.sum(dim=1) / lengths
        if not training:
            return nll
        options = self.cfg.get('loss', {'type': 'ce'})
        if options['type'] in ('focal', 'ce_focal_birth', 'birth_focal', 'birth_focal_split', 'birth_focal_pcgrad'):
            # -expm1(-CE) = 1 - p_gold, stable even when p_gold is near one.
            # Keep this factor in the graph: detaching it is not focal loss.
            gamma = options['gamma']
            modulation = (torch.ones_like(ce) if gamma == 0 else
                          (-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny).pow(gamma))
            token_loss = options['alpha'] * modulation * ce
            objective = token_loss.masked_fill(~valid, 0).sum(dim=1) / lengths
        else:
            objective = nll
        # Joint training needs the original CE gradient as well as the focal one.
        return torch.stack((objective, nll if options['type'] in ('ce_focal_birth', 'birth_focal', 'birth_focal_split', 'birth_focal_pcgrad') else nll.detach()), dim=1)

    def answer_loss(self, context, qa, *, training=False):
        q, target = self.question_and_target(qa)
        inputs = q + target[:-1]
        logits, _ = self.forward(ids=inputs, layers=context.layers, position_start=context.next_position,
                                  logits_slice=slice(len(q) - 1, len(q) - 1 + len(target)))
        return self._mean_answer_losses(logits, self.tensor(target), training=training)[0]

    def record_loss(self, record, *, vectors=None, prompt_ids=None):
        if self.cfg.get('loss', {}).get('type') in ('focal', 'ce_focal_birth', 'birth_focal', 'birth_focal_split', 'birth_focal_pcgrad'):
            return self.record_loss_components(record, vectors=vectors, prompt_ids=prompt_ids)[0]
        context = self.context(record, vectors=vectors, prompt_ids=prompt_ids)
        questions = questions_for(record, self.cfg)
        return grouped_loss(self.answer_losses(context, questions), questions)

    def record_loss_components(self, record, *, vectors=None, prompt_ids=None, return_partition=False, return_fields=False):
        context = self.context(record, vectors=vectors, prompt_ids=prompt_ids)
        questions = questions_for(record, self.cfg)
        values = self.answer_losses(context, questions, training=True)
        objective = grouped_loss(values[:, 0], questions)
        ce = grouped_loss(values[:, 1], questions)
        options = self.cfg.get('loss', {'type': 'ce'})
        if options['type'] in ('ce_focal_birth', 'birth_focal', 'birth_focal_split', 'birth_focal_pcgrad'):
            selected = torch.tensor([q.get('group', q['field']) == 'birth_date'
                                     or q['field'] in ('birth_year', 'birth_month', 'birth_day')
                                     for q in questions], device=values.device)
            # Keep the original field weight: extra = focal_birth / available_field_count.
            extra = grouped_loss(values[:, 0].masked_fill(~selected, 0), questions)
            birth = values[selected, 0].mean() if selected.any() else extra
            if options['type'] in ('birth_focal', 'birth_focal_split', 'birth_focal_pcgrad'):
                other = grouped_loss(values[:, 1].masked_fill(selected, 0), questions)
                result = (other + extra, {'ce_loss': ce.detach(), 'ce_other_loss': other.detach(),
                                         'focal_loss': extra.detach(), 'birth_date_focal_loss': birth.detach()})
                if return_fields:
                    groups = [q.get('group', q['field']) for q in questions]
                    mixed = torch.where(selected, values[:, 0], values[:, 1])
                    # Zero all other groups, but keep the original field-count
                    # denominator: sum(fields.values()) is exactly other+extra.
                    fields = {field: grouped_loss(mixed.masked_fill(
                        torch.tensor([g != field for g in groups], device=values.device), 0), questions)
                              for field in sorted(set(groups))}
                    return *result, fields
                # Both objectives share the same encoding/answer forward graph.
                # Training extracts each objective's gradient for its assigned U segment.
                return (*result, (other, extra)) if return_partition else result
            weighted = options['weight'] * extra
            return ce + weighted, {'ce_loss': ce.detach(), 'focal_loss': extra.detach(),
                                   'birth_date_focal_loss': birth.detach(),
                                   'focal_weighted_loss': weighted.detach()}
        return objective, {'ce_loss': ce.detach(), 'focal_loss': objective.detach()}

    def record_contrastive_loss(self, record, negatives, *, vectors):
        from torch.utils.checkpoint import checkpoint
        from .diagnosis_contrastive import ranking_loss
        options = self.cfg['diagnosis_contrastive']
        if not 1 <= len(negatives) <= options['max_negatives_per_record']:
            raise ValueError('Each training record needs one to four diagnosis negatives, without padding')
        context = self.context(record, vectors=vectors)
        questions = questions_for(record, self.cfg)
        losses = self.answer_losses(context, questions)
        ce = grouped_loss(losses, questions)
        index = next(i for i, q in enumerate(questions) if q['field'] == 'diagnosis')
        positive = losses[index]
        negative_losses = []
        size = options['negative_batch_size']
        for start in range(0, len(negatives), size):
            chunk = [dict(questions[index], answer=text) for text in negatives[start:start + size]]
            if options['checkpoint_negatives'] and torch.is_grad_enabled():
                flat = tuple(t for pair in context.layers for t in pair)
                # Bind the chunk: backward recomputation must not see the loop's last chunk.
                def calculate(*tensors, chunk=chunk):
                    ctx = replace(context, layers=tuple(zip(tensors[::2], tensors[1::2])))
                    return self.answer_losses(ctx, chunk)
                values = checkpoint(calculate, *flat, use_reentrant=False, preserve_rng_state=False)
            else:
                values = self.answer_losses(context, chunk)
            negative_losses.append(values)
        negative = torch.cat(negative_losses)
        if not torch.isfinite(losses).all() or not torch.isfinite(negative).all():
            raise FloatingPointError('Non-finite positive/negative diagnosis NLL')
        rank = ranking_loss(positive, negative, options['margin'])
        total = ce + options['weight'] * rank / 4
        return total, {'ce_loss': ce.detach(), 'diagnosis_ce_loss': positive.detach(),
                       'diagnosis_contrastive_loss': rank.detach(),
                       'diagnosis_negative_nll': negative.detach().mean(),
                       'diagnosis_negatives_per_record': positive.detach().new_tensor(float(len(negatives))),
                       'diagnosis_ranking_accuracy': (negative.detach() > positive.detach()).float().mean(),
                       'diagnosis_margin_accuracy': (negative.detach() - positive.detach() >= options['margin']).float().mean()}

    def _question_batch(self, context, sequences):
        """Left-pad independent queries; retain their unpadded absolute positions."""
        batch, width = len(sequences), max(map(len, sequences))
        lengths = torch.tensor([len(s) for s in sequences], device=self.device)
        columns = torch.arange(width, device=self.device)[None]
        valid = columns >= (width - lengths[:, None])
        ids = torch.full((batch, width), self.eos, device=self.device, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            ids[i, width - len(sequence):] = torch.tensor(sequence, device=self.device)
        positions = context.next_position + (columns - (width - lengths[:, None])).clamp_min(0)
        layers = tuple((k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in context.layers)
        key_valid = torch.cat((torch.ones((batch, context.physical_length), device=self.device, dtype=torch.bool), valid), dim=1)
        mask = causal_mask(width, context.physical_length, device=self.device, dtype=self.dtype).expand(batch, -1, -1, -1).clone()
        mask.masked_fill_(~key_valid[:, None, None, :], torch.finfo(self.dtype).min)
        return ids, lengths, positions, layers, mask, key_valid

    def answer_losses(self, context, questions, *, training=False):
        size = self.cfg.get('question_batch_size') or len(questions)
        chunks = [questions[i:i + size] for i in range(0, len(questions), size)]
        if self.cfg.get('checkpoint_questions', False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            flat = tuple(t for pair in context.layers for t in pair)
            def calculate(*tensors, chunk):
                ctx = replace(context, layers=tuple(zip(tensors[::2], tensors[1::2])))
                return self._answer_losses(ctx, chunk, training=training)
            return torch.cat([checkpoint(lambda *ts, chunk=chunk: calculate(*ts, chunk=chunk),
                                         *flat, use_reentrant=False, preserve_rng_state=False)
                              for chunk in chunks])
        return torch.cat([self._answer_losses(context, chunk, training=training) for chunk in chunks])

    def _answer_losses(self, context, questions, *, training=False):
        if not self.cfg.get('batch_questions', True):
            return torch.stack([self.answer_loss(context, qa, training=training) for qa in questions])
        tokenized = [self.question_and_target(qa) for qa in questions]
        sequences = [q + target[:-1] for q, target in tokenized]
        ids, _, positions, layers, mask, _ = self._question_batch(context, sequences)
        targets = [target for _, target in tokenized]
        width = max(map(len, targets))
        labels = torch.full((len(targets), width), -100, device=self.device, dtype=torch.long)
        for i, target in enumerate(targets):
            labels[i, :len(target)] = torch.tensor(target, device=self.device)
        lengths = torch.tensor([len(t) for t in targets], device=self.device)
        indices = (ids.shape[1] - lengths[:, None] + torch.arange(width, device=self.device)[None]).clamp_max(ids.shape[1] - 1)
        logits, _ = self.forward(ids=ids, layers=layers, position_ids=positions, mask=mask, logits_slice=indices)
        return self._mean_answer_losses(logits, labels, training=training)

    @torch.no_grad()
    def generate_many(self, context, questions):
        size = self.cfg.get('question_batch_size') or len(questions)
        return [result for i in range(0, len(questions), size)
                for result in self._generate_many(context, questions[i:i + size])]

    def _generate_many(self, context, questions):
        if not self.cfg.get('batch_questions', True):
            return [self.generate(context, qa) for qa in questions]
        sequences = [self.ids(self.question_text(qa['question'])) for qa in questions]
        ids, lengths, positions, layers, mask, key_valid = self._question_batch(context, sequences)
        logits, layers = self.forward(ids=ids, layers=layers, position_ids=positions, mask=mask, logits_slice=slice(-1, None))
        outputs = [[] for _ in questions]
        finished = [False] * len(questions)
        for step in range(self.cfg['max_new_tokens']):
            next_ids = logits[:, -1].argmax(dim=-1).tolist()
            for i, token in enumerate(next_ids):
                if not finished[i]:
                    if token in self.stops:
                        finished[i] = True
                    else:
                        outputs[i].append(token)
            if all(finished) or step + 1 == self.cfg['max_new_tokens']:
                break
            tokens = torch.tensor([[self.eos if done else token] for token, done in zip(next_ids, finished)], device=self.device)
            key_valid = torch.cat((key_valid, torch.ones((len(questions), 1), device=self.device, dtype=torch.bool)), dim=1)
            mask = torch.zeros((len(questions), 1, 1, key_valid.shape[1]), device=self.device, dtype=self.dtype)
            mask.masked_fill_(~key_valid[:, None, None, :], torch.finfo(self.dtype).min)
            logits, layers = self.forward(ids=tokens, layers=layers,
                    position_ids=(context.next_position + lengths + step)[:, None], mask=mask, logits_slice=slice(-1, None))
        return [(self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                 {'generated_tokens': len(tokens), 'stop_reason': 'eos' if done else 'max_new_tokens', 'truncated': not done})
                for tokens, done in zip(outputs, finished)]

    @torch.no_grad()
    def generate(self, context, qa):
        q = self.ids(self.question_text(qa['question']))
        logits, layers = self.forward(ids=q, layers=context.layers, position_start=context.next_position,
                                      logits_slice=slice(-1, None))
        ids, stop = [], 'max_new_tokens'
        for i in range(self.cfg['max_new_tokens']):
            token = int(logits[0, -1].argmax())
            if token in self.stops:
                stop = 'eos'
                break
            ids.append(token)
            if i + 1 < self.cfg['max_new_tokens']:
                logits, layers = self.forward(ids=[token], layers=layers,
                        position_start=context.next_position + len(q) + i, logits_slice=slice(-1, None))
        text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return text, {'generated_tokens': len(ids), 'stop_reason': stop, 'truncated': stop == 'max_new_tokens'}
