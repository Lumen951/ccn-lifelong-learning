"""Summarize run histories and plot learning/parameter diagnostics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_runs(patterns):
    directories = set()
    for pattern in patterns:
        directories.update(path for path in (ROOT / 'results').glob(pattern) if path.is_dir())
    runs = []
    for path in sorted(directories):
        if not (path / 'run.json').exists() or not (path / 'metrics.csv').exists():
            continue
        document = json.loads((path / 'run.json').read_text(encoding='utf-8'))
        with (path / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
            rows = [{key: float(value) if value else np.nan for key, value in row.items()}
                    for row in csv.DictReader(stream)]
        if rows:
            runs.append((path, document, rows))
    if not runs:
        raise ValueError('No matching runs with metrics')
    # Equal seeds and environment settings must result in identical streams.
    streams = defaultdict(set)
    for _, document, _ in runs:
        if document['status'] != 'completed':
            continue
        config = document['config']
        key = tuple((k, str(config[k])) for k in ('seed', 'steps', 'isi', 'iti', 'gamma'))
        streams[key].add(document['stream_sha256'])
    if any(len(hashes) != 1 for hashes in streams.values()):
        raise ValueError('Different architectures received different environment streams')
    return runs


def summarize(runs):
    summary = []
    for path, document, rows in runs:
        config, final = document['config'], rows[-1]
        changes = {'core_weight_change': None, 'head_weight_change': None}
        initial = path / 'checkpoint_000000000.pt'
        latest = path / f"checkpoint_{document['next_step']:09d}.pt"
        if config['model'] != 'ccn' and initial.exists() and latest.exists():
            first = torch.load(initial, map_location='cpu', weights_only=True)['learner']['model']
            last = torch.load(latest, map_location='cpu', weights_only=True)['learner']['model']
            for group, prefix in (('head_weight_change', True), ('core_weight_change', False)):
                squared = sum((last[key] - value).square().sum() for key, value in first.items()
                              if key.startswith('head.') == prefix)
                changes[group] = float(squared.sqrt())
        summary.append({'run': path.name, 'model': config['model'], 'seed': config['seed'],
                        'lr': config['lr'], 'context': config['context'], 'hidden': config['hidden'],
                        'features_per_stage': config['features_per_stage'],
                        'steps_per_stage': config['steps_per_stage'],
                        'normalization_rate': config['normalization_rate'],
                        'layers': config['layers'], 'status': document['status'],
                        'source_sha256': document['source_sha256'],
                        'steps': document['next_step'], 'parameters': document['parameter_count'],
                        'device': document['device'], 'mse': final['mse'], 'zero_mse': final['zero_mse'],
                        'mean_baseline_mse': final['mean_baseline_mse'],
                        'mse_over_zero': final['mse_over_zero'],
                        'valid_wait_mse': final['valid_wait_mse'],
                        'invalid_wait_mse': final['invalid_wait_mse'],
                        'prediction_gap': final['valid_prediction'] - final['invalid_prediction'],
                        'forget_mean': final['forget_mean'],
                        'outside_fraction': final['outside_fraction'], 'clip_fraction': final['clip_fraction'],
                        'elapsed_seconds': document['elapsed_seconds'],
                        'steps_per_second': document['next_step'] / max(document['elapsed_seconds'], 1e-9),
                        'peak_cuda_bytes': document.get('peak_cuda_bytes', 0), **changes})
    return summary


def plot_histories(runs, destination):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [('mse', 'Rolling prediction MSE'), ('mse_over_zero', 'MSE / zero baseline'),
              ('parameter_norm', 'Parameter norm (size dependent)'),
              ('td_rms', 'TD error RMS'), ('forget_mean', 'Forget gate mean (sampled)'),
              ('parameter_change', 'Parameter change since previous log')]
    colors = plt.cm.tab20(np.linspace(0, 1, len(runs)))
    for color, (path, document, rows) in zip(colors, runs):
        steps = [row['step'] for row in rows]
        for axis, (key, title) in zip(axes.flat, panels):
            axis.plot(steps, [row[key] for row in rows], label=path.name, color=color, linewidth=1.4)
            axis.set(title=title, xlabel='Environment steps')
            axis.grid(alpha=.2)
    axes[0, 1].axhline(1, color='black', linestyle='--', linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=min(3, len(runs)), fontsize=8)
    figure.suptitle('Online TD(lambda) pilot - unequal model sizes, not a compute-matched benchmark')
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_profiles(runs, destination):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for path, document, _ in runs:
        groups = defaultdict(list)
        start = max(0, document['next_step'] - document['config']['window'])
        with (path / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if int(row['step']) >= start and row['phase'] in ('valid_wait', 'invalid_wait'):
                    groups[(row['phase'], int(row['age']) // 5)].append(
                        (float(row['prediction']), float(row['target'])))
        for axis, phase in zip(axes, ('valid_wait', 'invalid_wait')):
            bins = sorted(bin_id for kind, bin_id in groups if kind == phase)
            if bins:
                axis.plot([5 * b + 2 for b in bins],
                          [np.mean(groups[(phase, b)], axis=0)[0] for b in bins], label=path.name)
        # Target is an evaluation-only oracle, not a training signal.
        if path == runs[0][0]:
            bins = sorted(b for kind, b in groups if kind == 'valid_wait')
            axes[0].plot([5 * b + 2 for b in bins],
                         [np.mean(groups[('valid_wait', b)], axis=0)[1] for b in bins],
                         color='black', linestyle='--', label='Evaluation target (sampled)')
    for axis, title in zip(axes, ('After valid cue', 'After invalid cue')):
        axis.set(title=title, xlabel='Steps since cue', ylabel='Mean prediction')
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_report(summary, output):
    lines = ['# Online trace-patterning comparison', '',
             'Pilot results, not reproductions of original-paper scores. Final MSE uses each run\'s last rolling window.',
             'Architectures share observations, reward, TD(lambda), and one update per step; parameter counts and learning rules differ.',
             'Time includes diagnostics/checkpoint overhead on this device. GPU and CPU times are not compute-matched.',
             'Pilot jobs may overlap: these timings are runtime estimates, not isolated speed benchmarks.', '',
             '| Run | Parameters | Final MSE | Zero MSE | Ratio | Valid-invalid prediction gap | Seconds |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append(f"| {row['run']} | {row['parameters']} | {row['mse']:.6f} | "
                     f"{row['zero_mse']:.6f} | {row['mse_over_zero']:.3f} | "
                     f"{row['prediction_gap']:.4f} | {row['elapsed_seconds']:.1f} |")
    groups = defaultdict(list)
    for row in summary:
        if row['status'] == 'completed':
            key = tuple(row[k] for k in ('model', 'lr', 'context', 'hidden', 'layers', 'steps',
                                        'features_per_stage', 'steps_per_stage', 'normalization_rate'))
            groups[key].append(row)
    lines.extend(['', 'Individual seeds and configurations above are the primary record.',
                  'Read the profiles alongside total MSE: a constant predictor can reduce MSE without remembering cues.',
                  'Forget gates near one can retain state longer, but alone do not prove useful memory.',
                  'Parameter norms depend on model size; parameter changes and gradients are diagnostics, not rankings.', ''])
    lines.extend(['## 参数变化', '',
                  '以下是初始到最终快照的参数差值范数；它不等于全部更新长度，也不直接代表预测能力。', '',
                  '| Run | 特征提取部分变化 | 输出层变化 |', '| --- | ---: | ---: |'])
    for row in summary:
        if row['core_weight_change'] is not None:
            lines.append(f"| {row['run']} | {row['core_weight_change']:.6f} | {row['head_weight_change']:.6f} |")
    worse = sum(row['mse'] >= row['mean_baseline_mse'] for row in summary)
    lines.extend(['', f'本次所选 {len(summary)} 组运行中，{worse} 组末窗口误差没有优于历史均值基线。',
                  '因此不能只凭优于零预测就认定学会了延迟关联。请同时检查有效／无效线索的预测差异和曲线。', ''])
    for key, rows in groups.items():
        if len(rows) > 1 and len({row['seed'] for row in rows}) == len(rows):
            values = np.array([row['mse'] for row in rows])
            lines.append(f'- {key}: {len(rows)} seeds, final MSE mean {values.mean():.6f}, '
                         f'standard deviation {values.std(ddof=1):.6f}.')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Directory names/globs under results/')
    parser.add_argument('--output', default='comparison', help='Output filename stem under results/')
    args = parser.parse_args()
    if not args.output or any(c in args.output for c in '/\\:<>"|?*') or args.output in ('.', '..'):
        parser.error('output must be a single filename stem')
    try:
        runs = load_runs(args.runs)
        summary = summarize(runs)
        base = ROOT / 'results' / args.output
        with base.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        plot_histories(runs, base.with_suffix('.png'))
        plot_profiles(runs, base.with_name(base.name + '_profiles').with_suffix('.png'))
        write_report(summary, base.with_suffix('.md'))
        print(base.with_suffix('.md'))
    except (ValueError, OSError, KeyError) as exc:
        print(f'Comparison failed: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
