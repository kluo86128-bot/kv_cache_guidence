"""Headless loss curves from scalar logs; no model or RNG operations."""
import json
import os
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class LossCurve:
    def __init__(self, directory, *, max_step=None, max_epoch=None, implementation_only=False):
        self.directory = Path(directory)
        self.implementation_only = implementation_only
        self.training = self._history('training.jsonl', max_step, None)
        self.validation = self._history('validation.jsonl', max_step, max_epoch)

    def _history(self, filename, max_step, max_epoch):
        path = self.directory / filename
        rows = {}
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if max_step is not None and row['step'] > max_step:
                    continue
                if max_epoch is not None and row['epoch'] > max_epoch:
                    continue
                rows[row['step']] = row
        return [rows[step] for step in sorted(rows)]

    def update_training(self, row):
        self.training.append(row)
        if row['step'] == 1 or row['step'] % 10 == 0:
            self.save()

    def update_validation(self, row):
        self.validation.append(row)
        self.save()

    def save(self):
        if not self.training:
            return None
        figure = Figure(figsize=(10, 5.5), dpi=150, facecolor='white')
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        steps = [row['step'] for row in self.training]
        values = [row['loss'] for row in self.training]
        smoothed = [sum(values[max(0, i - 19):i + 1]) / min(20, i + 1) for i in range(len(values))]
        contrastive = any('diagnosis_contrastive_loss' in row for row in self.training)
        focal = any(row.get('loss_type') == 'focal' for row in self.training)
        joint = any(row.get('loss_type') == 'ce_focal_birth' for row in self.training)
        birth_focal = any(row.get('loss_type') in ('birth_focal', 'birth_focal_split', 'birth_focal_pcgrad', 'field_focal_split', 'field_focal_pcgrad') for row in self.training)
        partitioned = any(row.get('loss_type') == 'birth_focal_split' for row in self.training)
        pcgrad = any(row.get('loss_type') == 'birth_focal_pcgrad' for row in self.training)
        field_pcgrad = any(row.get('loss_type') == 'field_focal_pcgrad' for row in self.training)
        field_partitioned = any(row.get('loss_type') == 'field_focal_split' for row in self.training)
        ax.plot(steps, values, color='#93b4de', alpha=.65, linewidth=1, marker='.', markersize=3,
                label='PCGrad: CE / selected-field Focal (unprojected objective)' if field_pcgrad else
                      'Routed other-field CE / selected-field Focal (sum for logging)' if field_partitioned else
                      'PCGrad: CE / birthday Focal (unprojected objective)' if pcgrad else
                      'Routed CE / birthday Focal (sum for logging)' if partitioned else
                      'Training other-field CE / birth-date Focal' if birth_focal else
                      'Training CE + birth-date Focal' if joint else
                      ('Training batch Focal' if focal else ('Training batch objective' if contrastive else 'Training batch NLL')))
        ax.plot(steps, smoothed, color='#215eaa', linewidth=2, marker='.' if len(steps) == 1 else None,
                label='Training trailing mean (up to 20 updates)')
        if contrastive or focal or joint or birth_focal:
            ce = [row for row in self.training if 'ce_loss' in row]
            ax.plot([r['step'] for r in ce], [r['ce_loss'] for r in ce], color='#24846a',
                    linewidth=1.4, linestyle='--', label='Training CE (field-averaged NLL)')
        if self.validation:
            ax.plot([row['step'] for row in self.validation],
                    [row['metrics']['macro_nll'] for row in self.validation],
                    color='#d56a21', linewidth=2, marker='o', markersize=5, label='Validation NLL (evaluated checkpoints)')
        suffix = ' | Random tiny model - implementation check only' if self.implementation_only else ''
        ax.set_title('Shared soft prompt: loss history' + suffix, fontsize=11, pad=14)
        ax.set_xlabel('Optimizer step')
        ax.set_ylabel('Training objective / answer NLL' if contrastive or focal or joint or birth_focal else 'Field-averaged answer NLL')
        ax.grid(True, color='#e4e8ed', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='best', frameon=False, fontsize=9)
        figure.text(.5, .025, 'Training: before updates; validation: evaluated checkpoints. Compare matching loss definitions.',
                    ha='center', fontsize=9, color='#536273')
        figure.tight_layout(rect=(0, .055, 1, 1))
        path = self.directory / 'loss_curve.png'
        temporary = path.with_name('loss_curve.tmp.png')
        figure.savefig(temporary, format='png')
        os.replace(temporary, path)
        if contrastive:
            self._save_contrastive_curve()
        return path

    def _save_contrastive_curve(self):
        rows = [r for r in self.training if 'diagnosis_contrastive_loss' in r]
        figure = Figure(figsize=(10, 5.5), dpi=150, facecolor='white')
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.plot([r['step'] for r in rows], [r['diagnosis_contrastive_loss'] for r in rows],
                color='#215eaa', label='Diagnosis contrastive hinge loss')
        ax.set_xlabel('Optimizer step')
        ax.set_ylabel('Mean hinge loss')
        accuracy = ax.twinx()
        accuracy.plot([r['step'] for r in rows], [r['diagnosis_ranking_accuracy'] for r in rows],
                      color='#24846a', linestyle='--', label='Positive ranks above negatives')
        accuracy.plot([r['step'] for r in rows], [r['diagnosis_margin_accuracy'] for r in rows],
                      color='#d56a21', linestyle=':', label='Required margin satisfied')
        accuracy.set_ylabel('Patient-averaged negative comparison fraction')
        accuracy.set_ylim(-.02, 1.02)
        ax.grid(True, color='#e4e8ed')
        lines = ax.get_lines() + accuracy.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc='best', frameon=False, fontsize=8)
        suffix = ' | Random tiny model - implementation check only' if self.implementation_only else ''
        ax.set_title('Diagnosis contrastive training' + suffix, fontsize=11)
        figure.tight_layout()
        path = self.directory / 'diagnosis_contrastive_curve.png'
        temporary = path.with_name('diagnosis_contrastive_curve.tmp.png')
        figure.savefig(temporary, format='png')
        os.replace(temporary, path)
