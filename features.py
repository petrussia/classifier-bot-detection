"""Признаки для каждой куки, рассчитанные строго внутри окна наблюдения.
Метки, абсолютные даты, идентификаторы куки и исходные ID объявлений не входят в матрицу.
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
EVENTS = [
    'search_results_view',
    'item_view',
    'photo_swipe',
    'seller_page_view',
    'contact_phone_show',
    'contact_chat_open',
    'contact_message_sent',
    'favorite_add',
    'login',
    'captcha_shown',
]
EV_CODE = {x: i for i, x in enumerate(EVENTS)}
PLATFORM_MAP = {'desktop': 'web', 'web': 'web', 'android': 'android', 'iphone': 'ios', 'ios': 'ios'}
TRANSITIONS = [(0, 1), (1, 0), (1, 2), (1, 4), (1, 7), (2, 1), (0, 0), (1, 1), (9, 0), (9, 1), (4, 1)]


def entropy_counts(counts):
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if not len(c):
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def stats(a, prefix, out, quantiles=True):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        for name in ['mean', 'std', 'min', 'max'] + (['q10', 'q25', 'median', 'q75', 'q90'] if quantiles else []):
            out[prefix + '_' + name] = np.nan
        return
    for name, v in [('mean', a.mean()), ('std', a.std()), ('min', a.min()), ('max', a.max())]:
        out[prefix + '_' + name] = v
    if quantiles:
        for name, v in zip(['q10', 'q25', 'median', 'q75', 'q90'], np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])):
            out[prefix + '_' + name] = v


def diversity(a, prefix, out):
    a = np.asarray(a)
    a = a[np.isfinite(a)] if a.dtype.kind in 'if' else a[pd.notna(a)]
    n = len(a)
    if not n:
        for name in [
            'unique',
            'ratio',
            'entropy',
            'entropy_norm',
            'top_share',
            'simpson',
            'singletons',
            'repeat_share',
            'switch_rate',
        ]:
            out[prefix + '_' + name] = 0.0
        return
    c = np.asarray(list(Counter(a).values()))
    h = entropy_counts(c)
    for name, v in [
        ('unique', len(c)),
        ('ratio', len(c) / n),
        ('entropy', h),
        ('entropy_norm', h / np.log(n) if n > 1 else 0),
        ('top_share', c.max() / n),
        ('simpson', np.sum((c / n) ** 2)),
        ('singletons', np.mean(c == 1)),
        ('repeat_share', c[c > 1].sum() / n),
        ('switch_rate', np.mean(a[1:] != a[:-1]) if n > 1 else 0),
    ]:
        out[prefix + '_' + name] = v


def ua_features(ua):
    s = str(ua).lower()
    if 'headless' in s:
        family = 'headless_chrome'
    elif 'yabrowser' in s:
        family = 'yandex'
    elif 'firefox' in s:
        family = 'firefox'
    elif s.startswith('avito/'):
        family = 'avito_app'
    elif 'chrome/' in s:
        family = 'chrome'
    elif 'safari/' in s:
        family = 'safari'
    elif 'python' in s or 'requests' in s:
        family = 'python'
    elif 'curl' in s:
        family = 'curl'
    else:
        family = 'other'
    if 'android' in s:
        os = 'android'
    elif 'iphone' in s or 'ipad' in s:
        os = 'ios'
    elif 'windows' in s:
        os = 'windows'
    elif 'macintosh' in s:
        os = 'macos'
    elif 'linux' in s or 'x11' in s:
        os = 'linux'
    else:
        os = 'other'
    return (family, os)


def _add_event_counts(cookie: dict, out: dict) -> None:
    """Частоты действий, контактные отношения и клиент."""
    e = cookie['code']
    n = len(e)
    cnt = np.bincount(e, minlength=10)
    for k, name in enumerate(EVENTS):
        out['n_' + name] = cnt[k]
        out['share_' + name] = cnt[k] / n
    out['n_event_types'] = np.count_nonzero(cnt)
    out['event_entropy'] = entropy_counts(cnt)
    out['event_max_share'] = cnt.max() / n
    out['contact_total'] = sum(cnt[4:7])
    out['contact_share'] = sum(cnt[4:7]) / n
    for name, num, den in [
        ('item_to_search', cnt[1], cnt[0]),
        ('contact_to_item', sum(cnt[4:7]), cnt[1]),
        ('photos_to_item', cnt[2], cnt[1]),
        ('favorites_to_item', cnt[7], cnt[1]),
        ('phone_to_chat', cnt[4], cnt[5] + cnt[6]),
    ]:
        out[name] = num / (1 + den)
    for col in ['platform_norm', 'browser', 'os']:
        c = Counter(cookie[col])
        out[col + '_mode'] = c.most_common(1)[0][0]
        out[col + '_unique'] = len(c)
        out[col + '_top_share'] = max(c.values()) / n
    out['ua_unique'] = len(set(cookie['user_agent']))
    for v in ['headless_chrome', 'avito_app', 'python', 'curl', 'chrome', 'firefox', 'yandex', 'safari']:
        out['browser_' + v + '_share'] = np.mean(cookie['browser'] == v)
    for v in ['web', 'android', 'ios']:
        out['platform_' + v + '_share'] = np.mean(cookie['platform_norm'] == v)


def _add_diversity(cookie: dict, out: dict) -> None:
    """Повторы объявлений, запросов и категорий без самих идентификаторов."""
    e = cookie['code']
    for col in ['item_id', 'item_category', 'item_location', 'search_query', 'seller_type']:
        vals = cookie[col]
        diversity(vals, col, out)
        out[col + '_missing_share'] = np.mean(pd.isna(vals))
    sellers = cookie['seller_type']
    out['pro_share'] = np.sum(sellers == 'pro') / max(1, np.sum(pd.notna(sellers)))
    for k, col in [
        (1, 'item_id'),
        (1, 'item_category'),
        (1, 'item_location'),
        (0, 'item_category'),
        (0, 'item_location'),
        (4, 'item_id'),
    ]:
        diversity(cookie[col][e == k], EVENTS[k] + '_' + col, out)


def _add_activity(t: np.ndarray, out: dict) -> None:
    """Распределение активности внутри окна наблюдения."""
    n = len(t)
    stats(t / 3600, 'hour', out, False)
    out['first_hour'] = t[0] / 3600
    out['last_hour'] = t[-1] / 3600
    out['span_s'] = t[-1] - t[0]
    out['events_per_active_min'] = n / (1 + out['span_s'] / 60)
    out['night_share'] = np.mean(t < 21600)
    out['evening_share'] = np.mean(t >= 64800)
    out['hour_sin'] = np.sin(t * 2 * np.pi / 86400).mean()
    out['hour_cos'] = np.cos(t * 2 * np.pi / 86400).mean()
    out['hour_concentration'] = np.hypot(out['hour_sin'], out['hour_cos'])
    for size in [60, 300, 1800, 3600]:
        _, counts = np.unique(np.floor_divide(t, size).astype(int), return_counts=True)
        for name, v in [
            ('active_bins', len(counts)),
            ('bin_max', counts.max()),
            ('bin_entropy', entropy_counts(counts)),
            ('bin_top_share', counts.max() / n),
        ]:
            out[f'{name}_{size}'] = v


def _add_gaps(t: np.ndarray, out: dict) -> None:
    """Длительность и регулярность пауз между действиями."""
    dt = np.diff(t)
    stats(dt, 'gap', out)
    mu = out['gap_mean']
    sd = out['gap_std']
    out['gap_cv'] = sd / (mu + 1e-06)
    out['gap_burstiness'] = (sd - mu) / (sd + mu + 1e-06)
    out['gap_iqr_relative'] = (out['gap_q75'] - out['gap_q25']) / (out['gap_median'] + 1)
    out['gap_max_relative'] = out['gap_max'] / (mu + 1)
    out['span_per_gap_median'] = out['span_s'] / (1 + out['gap_median'])
    for lim in [0, 1, 2, 5, 10, 30, 60, 120, 300, 900, 1800, 3600]:
        out[f'gap_le_{lim}'] = np.mean(dt <= lim) if len(dt) else np.nan
    if len(dt):
        vals, counts = np.unique(dt, return_counts=True)
        out['gap_unique_ratio'] = len(vals) / len(dt)
        out['gap_mode_share'] = counts.max() / len(dt)
        out['gap_mode'] = vals[counts.argmax()]
        out['gap_entropy'] = entropy_counts(counts)
        md = np.median(dt)
        out['gap_mad'] = np.median(np.abs(dt - md))
        out['gap_mad_relative'] = out['gap_mad'] / (1 + md)
        out['gap_within_20pct_median'] = np.mean(np.abs(dt - md) <= max(1, 0.2 * md))
        stats(np.log1p(dt), 'loggap', out, False)
        out['gap_successive_diff_abs_mean'] = np.abs(np.diff(dt)).mean() if len(dt) > 1 else np.nan
        out['gap_successive_diff_relative'] = out['gap_successive_diff_abs_mean'] / (mu + 1)
        out['gap_lag1_corr'] = (
            np.corrcoef(dt[:-1], dt[1:])[0, 1]
            if len(dt) > 3 and np.std(dt[:-1]) > 0 and np.std(dt[1:]) > 0
            else 0.0
        )
    else:
        for key in [
            'gap_unique_ratio',
            'gap_mode_share',
            'gap_mode',
            'gap_entropy',
            'gap_mad',
            'gap_mad_relative',
            'gap_within_20pct_median',
            'loggap_mean',
            'loggap_std',
            'loggap_min',
            'loggap_max',
            'gap_successive_diff_abs_mean',
            'gap_successive_diff_relative',
            'gap_lag1_corr',
        ]:
            out[key] = np.nan


def _add_sessions(t: np.ndarray, e: np.ndarray, out: dict) -> None:
    """Сессии при границах простоя 1, 5 и 30 минут."""
    n = len(t)
    dt = np.diff(t)
    for threshold in [60, 300, 1800]:
        stats(dt[dt <= threshold], f'shortgap_{threshold}', out, False)
        out[f'shortgap_{threshold}_cv'] = (
            out[f'shortgap_{threshold}_std'] / (1e-06 + out[f'shortgap_{threshold}_mean'])
        )
        split = np.r_[0, np.flatnonzero(dt > threshold) + 1, n]
        lengths = np.diff(split)
        durations = t[split[1:] - 1] - t[split[:-1]]
        out[f'sessions_{threshold}'] = len(lengths)
        out[f'session_{threshold}_max_events'] = lengths.max()
        out[f'session_{threshold}_mean_events'] = lengths.mean()
        out[f'session_{threshold}_longest_s'] = durations.max()
        out[f'session_{threshold}_total_s'] = durations.sum()
    for k in [0, 1, 4]:
        d = np.diff(t[e == k])
        stats(d, EVENTS[k] + '_gap', out)
        out[EVENTS[k] + '_gap_cv'] = out[EVENTS[k] + '_gap_std'] / (1e-06 + out[EVENTS[k] + '_gap_mean'])


def _add_transitions(e: np.ndarray, dt: np.ndarray, out: dict) -> None:
    """Частоты переходов и длины серий; переходы внутри секунды пропускаются."""
    n = len(e)
    if n > 1:
        # Переходы учитываются только между строго различными моментами.
        good = dt > 0
        transitions = e[:-1][good] * 10 + e[1:][good]
        cc = np.bincount(transitions, minlength=100)
        for k in range(10):
            out['self_transition_' + EVENTS[k]] = cc[k * 10 + k] / max(1, cc[k * 10:k * 10 + 10].sum())
        for i, j in TRANSITIONS:
            out[f'trans_{i}_{j}'] = cc[i * 10 + j] / max(1, len(transitions))
        out['event_switch_rate'] = np.mean(e[:-1][good] != e[1:][good]) if good.any() else 0
        out['transition_entropy'] = entropy_counts(cc)
        runs = np.diff(np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1, n])
        out['event_run_max'] = runs.max()
        out['event_run_mean'] = runs.mean()
    else:
        for name in EVENTS:
            out['self_transition_' + name] = 0
        for i, j in TRANSITIONS:
            out[f'trans_{i}_{j}'] = 0
        out['event_switch_rate'] = 0
        out['transition_entropy'] = 0
        out['event_run_max'] = 1
        out['event_run_mean'] = 1


def _add_search(cookie: dict, out: dict) -> None:
    """Глубина выдачи, шаги страниц и длина запроса."""
    pages = cookie['search_page']
    pages = pages[np.isfinite(pages)]
    stats(pages, 'page', out)
    out['page_unique'] = len(np.unique(pages))
    out['page_deep_share'] = np.mean(pages > 5) if len(pages) else 0
    if len(pages) > 1:
        d = np.diff(pages)
        out['page_step_plus1'] = np.mean(d == 1)
        out['page_step_positive'] = np.mean(d > 0)
        out['page_step_same'] = np.mean(d == 0)
        out['page_step_abs_mean'] = np.abs(d).mean()
    else:
        for name in ['page_step_plus1', 'page_step_positive', 'page_step_same', 'page_step_abs_mean']:
            out[name] = np.nan
    q = cookie['search_query']
    q = q[pd.notna(q)]
    stats([len(str(x)) for x in q], 'query_length', out, False)


def _add_pointer(cookie: dict, t: np.ndarray, out: dict) -> None:
    """Координаты и перемещения; отсутствие координат не считается неподвижностью."""
    x = cookie['pointer_x']
    yy = cookie['pointer_y']
    mask = np.isfinite(x) & np.isfinite(yy)
    xx = x[mask]
    yy = yy[mask]
    pt = t[mask]
    out['pointer_count'] = len(xx)
    out['pointer_share'] = mask.mean()
    stats(xx, 'pointer_x', out, False)
    stats(yy, 'pointer_y', out, False)
    if len(xx):
        unique, c = np.unique(np.stack([xx, yy], axis=1), axis=0, return_counts=True)
        out['pointer_unique_ratio'] = len(unique) / len(xx)
        out['pointer_top_share'] = c.max() / len(xx)
        out['pointer_center_share'] = np.mean((np.abs(xx - 960) < 100) & (np.abs(yy - 540) < 100))
        out['pointer_edge_share'] = np.mean((xx < 20) | (xx > 1900) | (yy < 20) | (yy > 1060))
        out['pointer_integer_10_share'] = np.mean((xx % 10 == 0) & (yy % 10 == 0))
    else:
        for name in [
            'pointer_unique_ratio',
            'pointer_top_share',
            'pointer_center_share',
            'pointer_edge_share',
            'pointer_integer_10_share',
        ]:
            out[name] = np.nan
    if len(xx) > 1:
        dist = np.hypot(np.diff(xx), np.diff(yy))
        stats(dist, 'pointer_dist', out)
        out['pointer_stationary_share'] = np.mean(dist == 0)
        out['pointer_straightness'] = np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]) / (1 + dist.sum())
        valid = np.diff(pt) > 0
        stats(dist[valid] / np.diff(pt)[valid], 'pointer_speed', out, False)
    else:
        stats([], 'pointer_dist', out)
        stats([], 'pointer_speed', out, False)
        out['pointer_stationary_share'] = np.nan
        out['pointer_straightness'] = np.nan


def build_features(data_dir, out_dir=None):
    """Возвращает метаданные, базовые признаки и счётчики очистки данных."""
    start = time.time()
    data_dir = Path(data_dir)
    dates = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
    tr = pd.read_csv(data_dir / 'train.csv', parse_dates=dates)
    te = pd.read_csv(data_dir / 'test.csv', parse_dates=dates)
    assert tr.cookie_id.is_unique and te.cookie_id.is_unique
    assert not set(tr.cookie_id) & set(te.cookie_id)
    meta = pd.concat([tr.assign(split='train'), te.assign(split='test')], ignore_index=True)
    assert meta.cookie_id.is_unique and (meta.window_end_ts > meta.window_start_ts).all()
    ev = pd.read_csv(data_dir / 'events.csv.gz', parse_dates=['event_ts'])
    audit = {
        'train_rows': len(tr),
        'test_rows': len(te),
        'raw_events': len(ev),
        'positive_train': int(tr.target.sum()),
        'exact_duplicate_rows': int(ev.duplicated().sum()),
    }
    # Присоединение many-to-one: одно окно на cookie, размножение строк запрещено.
    ev = ev.merge(meta[['cookie_id'] + dates], on='cookie_id', how='left', validate='many_to_one')
    assert ev.window_start_ts.notna().all()
    # Левая граница включена, правая исключена. Будущие события недоступны модели.
    valid = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    audit['events_before_window'] = int((ev.event_ts < ev.window_start_ts).sum())
    audit['events_at_or_after_end'] = int((ev.event_ts >= ev.window_end_ts).sum())
    audit['events_within_window_before_dedup'] = int(valid.sum())
    ev = ev.loc[valid].copy()
    # Только полные дубли; разные события в одну секунду НЕ удаляются.
    dup = ev.duplicated()
    audit['duplicates_within_window'] = int(dup.sum())
    ev = ev.loc[~dup].copy()
    audit['clean_events'] = len(ev)
    audit['cookies_without_in_window_events'] = int(len(meta) - ev.cookie_id.nunique())
    audit['events_before_cookie_creation'] = int((ev.event_ts < ev.cookie_created_at).sum())
    # Регистр и синонимы платформ нормализуем. Из UA берём семейство и ОС, не версии.
    ev['platform_norm'] = ev.platform.str.strip().str.lower().map(PLATFORM_MAP).fillna('other')
    um = {v: ua_features(v) for v in ev.user_agent.unique()}
    ev['browser'] = ev.user_agent.map(lambda x: um[x][0])
    ev['os'] = ev.user_agent.map(lambda x: um[x][1])
    # Исходные строки перемешаны. Внутри секунды порядок технический, не причинный.
    ev = ev.sort_values([
        'cookie_id',
        'event_ts',
        'eid',
        'item_id',
        'search_page',
    ], kind='stable', na_position='last')
    ev['time_s'] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev['code'] = ev.event_name.map(EV_CODE)
    assert ev.code.notna().all()
    # Массивы и срезы вместо DataFrame.apply для ускорения обработки событий.
    arr = {c: ev[c].to_numpy() for c in [
        'cookie_id',
        'time_s',
        'code',
        'platform_norm',
        'browser',
        'os',
        'user_agent',
        'item_id',
        'item_category',
        'item_location',
        'seller_type',
        'search_query',
        'search_page',
        'pointer_x',
        'pointer_y',
    ]}
    ids = arr['cookie_id']
    cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ev)]
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for a, b in zip(cuts[:-1], cuts[1:]):
            cookie = {name: values[a:b] for name, values in arr.items()}
            t = cookie['time_s']
            e = cookie['code']
            f = {'cookie_id': ids[a], 'n_events': b - a}
            _add_event_counts(cookie, f)
            _add_diversity(cookie, f)
            _add_activity(t, f)
            _add_gaps(t, f)
            _add_sessions(t, e, f)
            _add_transitions(e, np.diff(t), f)
            _add_search(cookie, f)
            _add_pointer(cookie, t, f)
            records.append(f)
    X = pd.DataFrame(records).set_index('cookie_id').reindex(meta.cookie_id)
    X.index = meta.index
    # Возраст куки известен в пределах окна; сама дата создания в X не входит.
    X['cookie_age_days'] = (meta.window_start_ts - meta.cookie_created_at).dt.total_seconds() / 86400
    X['cookie_age_end_days'] = (meta.window_end_ts - meta.cookie_created_at).dt.total_seconds() / 86400
    X['window_hours'] = (meta.window_end_ts - meta.window_start_ts).dt.total_seconds() / 3600
    X['age_at_first_event_hours'] = X['cookie_age_days'] * 24 + X['first_hour']
    X = X.replace([np.inf, -np.inf], np.nan)
    cats = X.select_dtypes('object').columns.tolist()
    for c in cats:
        X[c] = X[c].fillna('missing').astype(str)
    for c in X.columns.difference(cats):
        X[c] = X[c].astype(np.float32)
    audit.update(
        feature_count=X.shape[1],
        categorical_features=cats,
        feature_build_seconds=round(time.time() - start, 2),
        train_date_min=str(tr.window_start_ts.min()),
        train_date_max=str(tr.window_start_ts.max()),
        test_date_min=str(te.window_start_ts.min()),
        test_date_max=str(te.window_start_ts.max()),
    )
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        X.to_pickle(out_dir / 'features.pkl')
        meta.to_pickle(out_dir / 'metadata.pkl')
        (out_dir / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f"Base features: {X.shape}; clean events: {len(ev)}; "
          f"seconds: {audit['feature_build_seconds']}", flush=True)
    return (meta, X, audit)
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='cache')
    a = p.parse_args()
    build_features(a.data_dir, a.out_dir)
