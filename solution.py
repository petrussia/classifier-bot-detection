"""Обучение ансамбля, временная проверка и запись submission.csv."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import Booster, LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from features import build_features
from features_extra import build_extra
from metric import precision_at_recall, recall_at_fpr

ROOT = Path(__file__).resolve().parent


def dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Метрики и диагностический порог. В submission остаются непрерывные score."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.shape != score.shape or not len(y) or len(np.unique(y)) != 2:
        raise ValueError('Для оценки нужны равные непустые массивы и оба класса.')
    if not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
        raise ValueError('Score должен быть конечным числом от 0 до 1.')
    order = np.argsort(-score, kind='mergesort')
    ys, ss = y[order], score[order]
    # Порог не должен разделять группу одинаковых score.
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    tp = np.cumsum(ys)[ends]
    counts = ends + 1
    precision = tp / counts
    recall = tp / y.sum()
    best = int(np.argmax(np.where(recall >= .7, precision, -np.inf)))
    return {
        'n_cookies': len(y),
        'n_bots': int(y.sum()),
        'positive_rate': float(y.mean()),
        'precision_at_recall_ge_0_70': precision_at_recall(y, score),
        'average_precision': float(average_precision_score(y, score)),
        'roc_auc': float(roc_auc_score(y, score)),
        'log_loss': float(log_loss(y, score)),
        'recall_at_fpr_0_01': recall_at_fpr(y, score),
        'diagnostic_threshold': float(ss[ends[best]]),
        'diagnostic_recall': float(recall[best]),
        'diagnostic_true_positives': int(tp[best]),
        'diagnostic_false_positives': int(counts[best] - tp[best]),
        'diagnostic_flagged': int(counts[best]),
    }


def categorical_matrix(X: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    """Неизвестные LightGBM категории становятся NaN; словарь берётся из train."""
    X = X.copy()
    for column, categories in vocab.items():
        X[column] = pd.Categorical(X[column], categories=categories)
    return X


def combine(member_scores: list[np.ndarray], config: dict) -> np.ndarray:
    weights = np.asarray([member['weight'] for member in config['members']], dtype=float)
    if (len(weights) != len(member_scores) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Проверьте веса и число моделей в ансамбле.')
    weights /= weights.sum()
    return np.average(np.asarray(member_scores), weights=weights, axis=0)


def make_model(member: dict, seed: int):
    """Все параметры обучения, кроме текущего seed, находятся в JSON."""
    if member['family'] == 'catboost':
        return CatBoostClassifier(**member['params'], random_seed=seed)
    if member['family'] == 'lightgbm':
        return LGBMClassifier(**member['params'], random_state=seed)
    raise ValueError(f"Неизвестная модель: {member['family']}")


def fit_predict(
    feature_sets: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    fit_mask: np.ndarray,
    predict_mask: np.ndarray,
    config: dict,
    model_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    """Обучает фиксированный ансамбль. Проверочные метки в fit не передаются."""
    y = meta.loc[fit_mask, 'target'].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError('В обучающей выборке должны быть оба класса.')
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    predictions, member_predictions, descriptions = [], {}, []
    for index, member in enumerate(config['members']):
        X = feature_sets[member['features']]
        # На суточных окнах возраст в конце дублирует возраст в начале.
        columns = [c for c in X if c != 'cookie_age_end_days'
                   and X.loc[fit_mask, c].nunique(dropna=False) > 1]
        X = X[columns]
        cats = X.select_dtypes('object').columns.tolist()
        vocab = {c: sorted(X.loc[fit_mask, c].unique().tolist()) for c in cats}
        family = member['family']
        if family == 'lightgbm':
            X = categorical_matrix(X, vocab)
        entry = {'member_index': index, 'member': member, 'columns': columns,
                 'categorical_columns': cats, 'vocabulary': vocab, 'files': []}
        seed_scores = []
        for seed in member['seeds']:
            started = time.time()
            model = make_model(member, seed)
            fit_args = {'cat_features': cats} if family == 'catboost' else {}
            model.fit(X.loc[fit_mask], y, **fit_args)
            score = model.predict_proba(X.loc[predict_mask])[:, 1]
            seed_scores.append(np.asarray(score, dtype=float))
            if model_dir is not None:
                suffix = 'cbm' if family == 'catboost' else 'txt'
                filename = f'member_{index:02d}_seed_{seed}.{suffix}'
                saved_model = model if family == 'catboost' else model.booster_
                saved_model.save_model(str(model_dir / filename))
                entry['files'].append(filename)
            print(f'{family}: seed={seed}, features={len(columns)}, '
                  f'seconds={time.time() - started:.1f}', flush=True)
        mean_score = np.mean(seed_scores, axis=0)
        predictions.append(mean_score)
        member_predictions[f'member_{index}_{family}'] = mean_score
        descriptions.append(entry)
    if model_dir is not None:
        dump_json({'config': config, 'members': descriptions}, model_dir / 'manifest.json')
    return combine(predictions, config), member_predictions, descriptions


def predict_saved(feature_sets: dict[str, pd.DataFrame], mask: np.ndarray,
                  model_dir: Path) -> np.ndarray:
    manifest_path = model_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} не найден. Сначала выполните --mode train.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    member_scores = []
    for entry in manifest['members']:
        member = entry['member']
        family = member['family']
        X = feature_sets[member['features']].loc[mask, entry['columns']]
        if family == 'lightgbm':
            X = categorical_matrix(X, entry['vocabulary'])
        runs = []
        for filename in entry['files']:
            path = model_dir / filename
            if family == 'catboost':
                model = CatBoostClassifier()
                model.load_model(str(path))
                score = model.predict_proba(X)[:, 1]
            elif family == 'lightgbm':
                score = Booster(model_file=str(path)).predict(X)
            else:
                raise ValueError(f'Неизвестная модель: {family}')
            runs.append(np.asarray(score, dtype=float))
        member_scores.append(np.mean(runs, axis=0))
    return combine(member_scores, manifest['config'])


def validate_submission(submission: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    if list(submission.columns) != ['cookie_id', 'score']:
        raise ValueError('В submission нужны только cookie_id и score.')
    if not submission.cookie_id.is_unique or len(submission) != len(test_meta):
        raise ValueError('В submission пропущены или повторяются строки.')
    if submission.cookie_id.tolist() != test_meta.cookie_id.tolist():
        raise ValueError('Идентификаторы или их порядок не совпадают с test.csv.')
    if not np.isfinite(submission.score).all() or not submission.score.between(0, 1).all():
        raise ValueError('Score должен быть конечным числом от 0 до 1.')


def write_submission(score: np.ndarray, test_meta: pd.DataFrame, path: Path) -> None:
    submission = pd.DataFrame({'cookie_id': test_meta.cookie_id.to_numpy(), 'score': score})
    validate_submission(submission, test_meta)
    # Не округляем score, чтобы не создавать новые совпадения при выборе порога.
    submission.to_csv(path, index=False, float_format='%.17g')
    reread = pd.read_csv(path, float_precision='round_trip')
    validate_submission(reread, test_meta)
    np.testing.assert_array_equal(reread.score.to_numpy(), score)
    print(f'Saved {path} ({len(submission)} rows), SHA-256: {sha256(path)}', flush=True)


def validate_holdout(feature_sets: dict, meta: pd.DataFrame, config: dict,
                     output_dir: Path) -> dict:
    train_mask = meta.split.eq('train').to_numpy()
    before = train_mask & meta.window_start_ts.lt(config['holdout_start']).to_numpy()
    holdout = train_mask & meta.window_start_ts.ge(config['holdout_start']).to_numpy()
    y = meta.loc[holdout, 'target'].astype(int).to_numpy()
    score, members, _ = fit_predict(feature_sets, meta, before, holdout, config)
    results = {'ensemble': evaluate(y, score)}
    results.update({name: evaluate(y, values) for name, values in members.items()})
    base = feature_sets['base']
    simple = ['n_events', 'item_id_unique']
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=4)
    rf.fit(base.loc[before, simple].fillna(0), meta.loc[before, 'target'].astype(int))
    baseline = rf.predict_proba(base.loc[holdout, simple].fillna(0))[:, 1]
    results['baseline_random_forest'] = evaluate(y, baseline)
    constant = np.full(len(y), meta.loc[before, 'target'].mean())
    results['constant'] = evaluate(y, constant)
    predictions = meta.loc[holdout, ['cookie_id', 'window_start_ts', 'target']].copy()
    predictions['score'] = score
    predictions['baseline_score'] = baseline
    for name, values in members.items():
        predictions[name] = values
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / 'holdout_predictions.csv', index=False)
    dump_json(results, output_dir / 'holdout_metrics.json')
    print('Holdout:', results['ensemble']['precision_at_recall_ge_0_70'], flush=True)
    return results


def run(data_dir: Path, output_dir: Path, config_path: Path, mode: str) -> None:
    if mode not in {'all', 'train', 'validate', 'predict'}:
        raise ValueError(f'Неизвестный режим: {mode}')
    for filename in ['train.csv', 'test.csv', 'events.csv.gz']:
        if not (data_dir / filename).is_file():
            raise FileNotFoundError(f'Нет файла {data_dir / filename}. Укажите --data-dir.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if mode == 'predict':
        # Наборы признаков при inference должны соответствовать сохранённым моделям.
        manifest = output_dir / 'models' / 'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'Нет {manifest}. Сначала выполните --mode train.')
        config = json.loads(manifest.read_text(encoding='utf-8'))['config']
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, base, audit = build_features(data_dir)
    feature_sets = {'base': base}
    if any(m['features'] == 'augmented' for m in config['members']):
        feature_sets['augmented'] = pd.concat([base, build_extra(data_dir, meta)], axis=1)
    dump_json(audit, output_dir / 'data_audit.json')
    train_mask = meta.split.eq('train').to_numpy()
    test_mask = ~train_mask
    if mode in {'all', 'validate'}:
        validate_holdout(feature_sets, meta, config, output_dir)
    if mode == 'validate':
        return
    if mode == 'predict':
        score = predict_saved(feature_sets, test_mask, output_dir / 'models')
    else:
        score, _, _ = fit_predict(feature_sets, meta, train_mask, test_mask,
                                  config, output_dir / 'models')
        reloaded = predict_saved(feature_sets, test_mask, output_dir / 'models')
        np.testing.assert_allclose(score, reloaded, rtol=0, atol=1e-14)
        dump_json({'max_abs_difference_after_reload': float(np.max(np.abs(score-reloaded)))},
                  output_dir / 'reload_check.json')
    dest = output_dir / 'submission.csv'
    write_submission(score, meta.loc[test_mask], dest)
    dump_json({
        'python': platform.python_version(),
        'submission_sha256': sha256(dest),
        'config_sha256': sha256(config_path),
        'train_rows': int(train_mask.sum()),
        'test_rows': int(test_mask.sum()),
        'input_sha256': {name: sha256(data_dir / name)
                         for name in ['train.csv', 'test.csv', 'events.csv.gz']},
    }, output_dir / 'reproducibility.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--config', type=Path, default=ROOT / 'final_config.json')
    parser.add_argument('--mode', choices=['all', 'validate', 'train', 'predict'], default='all')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.config, args.mode)


if __name__ == '__main__':
    main()
