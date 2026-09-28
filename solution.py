"""Воспроизводимый процесс обнаружения ботов без обращения к сети.

Запустите `python solution.py --mode all`, чтобы проверить зафиксированную конфигурацию,
переобучить модели на всех размеченных куки и создать artifacts/submission.csv.
"""
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
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, Booster

from features import build_features
from metric import precision_at_recall, recall_at_fpr

ROOT = Path(__file__).resolve().parent


def dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Официальная целевая метрика, диагностика и лучший допустимый порог.

    Порог используется только для ретроспективной диагностики на проверочной выборке:
    он не выбирается для файла предсказаний и не превращает оценки в бинарные метки.
    Одинаковые оценки обрабатываются целыми группами, как в metric.py.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    order = np.argsort(-score, kind='mergesort')
    ys, ss = y[order], score[order]
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    tp = np.cumsum(ys)[ends]
    counts = ends + 1
    precision = tp / counts
    recall = tp / y.sum()
    eligible = recall >= .7
    best = int(np.argmax(np.where(eligible, precision, -np.inf)))
    return {
        'n_cookies': len(y), 'n_bots': int(y.sum()),
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
    """Использует только категории обучающей выборки; неизвестные будущие категории заменяются на NaN."""
    X = X.copy()
    for col, categories in vocab.items():
        X[col] = pd.Categorical(X[col], categories=categories)
    return X


def combine(member_scores: list[np.ndarray], config: dict[str, Any]) -> np.ndarray:
    weights = np.asarray([m['weight'] for m in config['members']], dtype=float)
    if len(weights) != len(member_scores) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid ensemble specification')
    weights /= weights.sum()
    scores = np.asarray(member_scores)
    if config['blend'] == 'probability':
        return np.average(scores, weights=weights, axis=0)
    if config['blend'] == 'logit':
        return expit(np.average(logit(np.clip(scores, 1e-6, 1-1e-6)), weights=weights, axis=0))
    raise ValueError('Unknown blending method')


def fit_predict(
    feature_sets: dict[str, pd.DataFrame], meta: pd.DataFrame,
    fit_mask: np.ndarray, predict_mask: np.ndarray,
    config: dict[str, Any], model_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Обучает все модели с фиксированными гиперпараметрами и усредняет запуски с разными seed-ами.

    Метки проверочной и тестовой выборок не передаются в model.fit;
    ранняя остановка не применяется. Гиперпараметры уже зафиксированы в JSON.
    Постоянные столбцы и наборы категорий определяются только по обучающим строкам.
    """
    y = meta.loc[fit_mask, 'target'].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError('Training data must contain both target classes')
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
    predictions, member_predictions, descriptions = [], {}, []
    for member_index, member in enumerate(config['members']):
        X = feature_sets[member.get('features', 'base')]
        cols = [c for c in X if c != 'cookie_age_end_days' and X.loc[fit_mask, c].nunique(dropna=False) > 1]
        X = X[cols]
        cats = X.select_dtypes('object').columns.tolist()
        vocab = {c: sorted(X.loc[fit_mask, c].unique().tolist()) for c in cats}
        family = member['family']
        XX = X if family == 'catboost' else categorical_matrix(X, vocab)
        seed_predictions = []
        entry = {'member_index': member_index, 'member': member, 'columns': cols,
                 'categorical_columns': cats, 'vocabulary': vocab, 'files': []}
        for seed in member['seeds']:
            started = time.time()
            if family == 'catboost':
                model = CatBoostClassifier(
                    iterations=member['iterations'], depth=member.get('depth', 6),
                    learning_rate=.035, l2_leaf_reg=5, loss_function='Logloss',
                    random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False)
                model.fit(XX.loc[fit_mask], y, cat_features=cats)
                p = model.predict_proba(XX.loc[predict_mask])[:, 1]
                suffix = 'cbm'
            elif family == 'lightgbm':
                leaves = member.get('num_leaves', 31)
                model = LGBMClassifier(
                    n_estimators=member['iterations'], learning_rate=.03,
                    num_leaves=leaves, min_child_samples=30 if leaves == 31 else 35,
                    reg_lambda=5., colsample_bytree=.85, verbosity=-1, n_jobs=4,
                    random_state=seed, deterministic=True, force_col_wise=True)
                model.fit(XX.loc[fit_mask], y)
                p = model.predict_proba(XX.loc[predict_mask])[:, 1]
                suffix = 'txt'
            elif family == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=member['iterations'], learning_rate=.03,
                    max_depth=member.get('depth', 3), min_child_weight=5,
                    reg_lambda=10, subsample=.85, colsample_bytree=.85,
                    n_jobs=4, random_state=seed, tree_method='hist', enable_categorical=True)
                model.fit(XX.loc[fit_mask], y)
                p = model.predict_proba(XX.loc[predict_mask])[:, 1]
                suffix = 'ubj'
            else:
                raise ValueError(f'Unknown model family: {family}')
            seed_predictions.append(np.asarray(p, dtype=float))
            filename = f'member_{member_index:02d}_seed_{seed}.{suffix}'
            if model_dir:
                path = model_dir / filename
                if family == 'lightgbm':
                    model.booster_.save_model(str(path))
                else:
                    model.save_model(str(path))
                entry['files'].append(filename)
                if family == 'catboost':
                    importance = model.feature_importances_
                elif family == 'lightgbm':
                    importance = model.booster_.feature_importance(importance_type='gain')
                else:
                    importance = model.feature_importances_
                pd.DataFrame({'feature': cols, 'importance': importance}).sort_values(
                    'importance', ascending=False).to_csv(model_dir / (filename + '.importance.csv'), index=False)
            print(f'{family}, seed={seed}, trees={member["iterations"]}, '
                  f'features={len(cols)}, seconds={time.time()-started:.1f}', flush=True)
        mean_p = np.mean(seed_predictions, axis=0)
        predictions.append(mean_p)
        member_predictions[f'member_{member_index}_{family}'] = mean_p
        descriptions.append(entry)
    p = combine(predictions, config)
    if model_dir:
        dump_json({'config': config, 'members': descriptions}, model_dir / 'manifest.json')
    return p, member_predictions, descriptions


def predict_saved(feature_sets: dict[str, pd.DataFrame], mask: np.ndarray,
                  model_dir: Path) -> np.ndarray:
    manifest = json.loads((model_dir / 'manifest.json').read_text(encoding='utf-8'))
    member_scores = []
    for entry in manifest['members']:
        member = entry['member']; family = member['family']
        X = feature_sets[member.get('features', 'base')].loc[mask, entry['columns']]
        if family != 'catboost':
            X = categorical_matrix(X, entry['vocabulary'])
        runs = []
        for filename in entry['files']:
            path = model_dir / filename
            if family == 'catboost':
                model = CatBoostClassifier(); model.load_model(str(path))
                p = model.predict_proba(X)[:, 1]
            elif family == 'lightgbm':
                model = Booster(model_file=str(path)); p = model.predict(X)
            else:
                from xgboost import XGBClassifier
                model = XGBClassifier(); model.load_model(str(path)); p = model.predict_proba(X)[:, 1]
            runs.append(np.asarray(p, dtype=float))
        member_scores.append(np.mean(runs, axis=0))
    return combine(member_scores, manifest['config'])


def validate_submission(submission: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    if list(submission.columns) != ['cookie_id', 'score']:
        raise ValueError('Submission must have exactly cookie_id and score columns')
    if not submission.cookie_id.is_unique or len(submission) != len(test_meta):
        raise ValueError('Duplicate or missing submission rows')
    if submission.cookie_id.tolist() != test_meta.cookie_id.tolist():
        raise ValueError('Submission cookie IDs or their order do not match test.csv')
    if not np.isfinite(submission.score).all() or not submission.score.between(0, 1).all():
        raise ValueError('Scores must be finite numbers between 0 and 1')


def run(data_dir: Path, output_dir: Path, config_path: Path, mode: str) -> None:
    config = json.loads(config_path.read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, base, audit = build_features(data_dir)
    feature_sets = {'base': base}
    if any(m.get('features', 'base') == 'augmented' for m in config['members']):
        from features_extra import build_extra
        feature_sets['augmented'] = pd.concat([base, build_extra(data_dir, meta)], axis=1)
    dump_json(audit, output_dir / 'data_audit.json')
    if mode == 'features':
        base.to_pickle(output_dir / 'features.pkl'); meta.to_pickle(output_dir / 'metadata.pkl')
        return
    train_mask = meta.split.eq('train').to_numpy()
    test_mask = ~train_mask
    if mode in ['all', 'validate']:
        before = train_mask & meta.window_start_ts.lt('2026-04-17').to_numpy()
        holdout = train_mask & meta.window_start_ts.ge('2026-04-17').to_numpy()
        # Граница разбиения и конфигурация моделей зафиксированы до оценки на отложенной выборке.
        y = meta.loc[holdout, 'target'].astype(int).to_numpy()
        p, members, desc = fit_predict(feature_sets, meta, before, holdout, config)
        results = {'ensemble': evaluate(y, p)}
        for name, values in members.items():
            results[name] = evaluate(y, values)
        simple = ['n_events', 'item_id_unique']
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=4)
        rf.fit(base.loc[before, simple].fillna(0), meta.loc[before, 'target'].astype(int))
        baseline = rf.predict_proba(base.loc[holdout, simple].fillna(0))[:, 1]
        results['baseline_random_forest'] = evaluate(y, baseline)
        constant = np.full(len(y), meta.loc[before, 'target'].mean())
        results['constant'] = evaluate(y, constant)
        dump_json(results, output_dir / 'holdout_metrics.json')
        holdout_predictions = meta.loc[holdout, ['cookie_id', 'window_start_ts', 'target']].copy()
        holdout_predictions['score'] = p
        holdout_predictions['baseline_score'] = baseline
        for name, values in members.items():
            holdout_predictions[name] = values
        holdout_predictions.to_csv(output_dir / 'holdout_predictions.csv', index=False)
        print('Holdout:', json.dumps(results['ensemble'], ensure_ascii=False), flush=True)
    if mode in ['all', 'train']:
        p, _, desc = fit_predict(feature_sets, meta, train_mask, test_mask, config, output_dir / 'models')
        # Проверяем сохранённые модели, а не только предсказания моделей в памяти.
        reloaded = predict_saved(feature_sets, test_mask, output_dir / 'models')
        if not np.allclose(p, reloaded, rtol=0, atol=1e-14):
            raise AssertionError('Serialized models do not reproduce predictions')
        dump_json({'max_abs_difference_after_reload': float(np.max(np.abs(p-reloaded)))},
                  output_dir / 'reload_check.json')
    elif mode == 'predict':
        p = predict_saved(feature_sets, test_mask, output_dir / 'models')
    if mode in ['all', 'train', 'predict']:
        test_meta = meta.loc[test_mask]
        sub = pd.DataFrame({'cookie_id': test_meta.cookie_id.to_numpy(), 'score': p})
        validate_submission(sub, test_meta)
        # Сохраняем точность. Округление и бинаризация оценок могут ухудшить ранжирование.
        dest = output_dir / 'submission.csv'
        sub.to_csv(dest, index=False, float_format='%.17g')
        reread = pd.read_csv(dest, float_precision='round_trip')
        validate_submission(reread, test_meta)
        assert np.array_equal(reread.score.to_numpy(), p)
        dump_json({'python': platform.python_version(), 'submission_sha256': sha256(dest),
                   'config_sha256': sha256(config_path), 'train_rows': int(train_mask.sum()),
                   'test_rows': int(test_mask.sum()), 'input_sha256': {
                       filename: sha256(data_dir / filename)
                       for filename in ['train.csv', 'test.csv', 'events.csv.gz']},
                   'unique_scores': int(sub.score.nunique()), 'finite_scores': True,
                   'score_min': float(sub.score.min()), 'score_max': float(sub.score.max())},
                  output_dir / 'reproducibility.json')
        print(f'Saved {dest} ({len(sub)} rows), SHA-256: {sha256(dest)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--config', type=Path, default=ROOT / 'final_config.json')
    parser.add_argument('--mode', choices=['all', 'validate', 'train', 'predict', 'features'], default='all')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.config, args.mode)


if __name__ == '__main__':
    main()
