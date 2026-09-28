"""Дополнительные признаки времени, навигации и геометрии движения указателя.
Используются только события внутри того же окна, без статистик между куками или по целевой переменной.
"""
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from features import stats, EV_CODE, EVENTS


def _add_trimmed_gaps(dt: np.ndarray, out: dict) -> None:
    """Регулярность без самых длинных пауз."""
    sd = np.sort(dt)
    for k in [1, 2, 3]:
        core = sd[:-k] if len(sd) > k else np.array([])
        stats(core, f'trim{k}_gap', out, False)
        out[f'trim{k}_cv'] = out[f'trim{k}_gap_std'] / (out[f'trim{k}_gap_mean'] + 1e-06)
    for frac in [0.5, 0.75, 0.9]:
        core = sd[:max(1, int(np.ceil(len(sd) * frac)))]
        key = 'gap_core' + str(int(100 * frac))
        stats(core, key, out, False)
        out[key + '_cv'] = out[key + '_std'] / (1e-06 + out[key + '_mean'])


def _add_gap_shape(dt: np.ndarray, out: dict) -> None:
    """Отношения квантилей и форма распределения коротких пауз."""
    for lim in [120, 300, 600, 1800]:
        d = dt[dt <= lim]
        key = f'gapsub_{lim}'
        stats(d, key, out)
        if len(d) > 1:
            md = np.median(d)
            mu = d.mean()
            ss = d.std()
            q = np.quantile(d, [0.1, 0.25, 0.75, 0.9])
            out[key + '_madrel'] = np.median(np.abs(d - md)) / (md + 1)
            out[key + '_q10median'] = q[0] / (1 + md)
            out[key + '_minmedian'] = d.min() / (1 + md)
            out[key + '_q90q10'] = (q[3] + 1) / (q[0] + 1)
            out[key + '_q75q25'] = (q[2] + 1) / (q[1] + 1)
            out[key + '_skew'] = np.mean(((d - mu) / (ss + 1e-06)) ** 3)
            out[key + '_kurt'] = np.mean(((d - mu) / (ss + 1e-06)) ** 4)
            ld = np.log1p(d)
            out[key + '_logmean'] = ld.mean()
            out[key + '_logstd'] = ld.std()
            out[key + '_geomean_arithmean'] = np.expm1(ld.mean()) / (mu + 1e-06)
        else:
            for name in [
                'madrel',
                'q10median',
                'minmedian',
                'q90q10',
                'q75q25',
                'skew',
                'kurt',
                'logmean',
                'logstd',
                'geomean_arithmean',
            ]:
                out[key + '_' + name] = np.nan


def _add_session_rates(t: np.ndarray, out: dict) -> None:
    """Размер, длительность и скорость сессий."""
    n = len(t)
    dt = np.diff(t)
    for lim in [120, 600, 1800]:
        cuts2 = np.r_[0, np.flatnonzero(dt > lim) + 1, n]
        lens = np.diff(cuts2)
        dur = t[cuts2[1:] - 1] - t[cuts2[:-1]]
        key = f'sessx_{lim}'
        stats(lens, key + '_size', out, False)
        stats(dur, key + '_duration', out, False)
        rates = (lens - 1) / (1 + dur)
        stats(rates, key + '_rate', out, False)
        out[key + '_single_share'] = np.mean(lens == 1)
        out[key + '_count'] = len(lens)


def _add_rolling_counts(t: np.ndarray, out: dict) -> None:
    """Пиковое число событий в скользящем окне."""
    n = len(t)
    # Скользящие окна избегают зависимости от произвольной границы минутных бинов.
    for width in [10, 30, 60, 120, 300, 900]:
        right = np.searchsorted(t, t + width, side='right')
        out[f'rolling_count_{width}_max'] = np.max(right - np.arange(n))


def _add_transition_matrix(e: np.ndarray, dt: np.ndarray, out: dict) -> None:
    """Матрица переходов и совпадения событий на лагах."""
    n = len(e)
    # Все пары типов действий; одинаковые timestamp исключены из переходов.
    if n > 1:
        good = dt > 0
        codes = e[:-1][good] * 10 + e[1:][good]
        co = np.bincount(codes, minlength=100) / max(1, len(codes))
    else:
        co = np.zeros(100)
    for i in range(10):
        for j in range(10):
            out[f'transall_{i}_{j}'] = co[i * 10 + j]
    for lag in [2, 3, 4]:
        out[f'event_equal_lag{lag}'] = np.mean(e[lag:] == e[:-lag]) if n > lag else np.nan


def _add_item_navigation(cookie: dict, e: np.ndarray, t: np.ndarray, out: dict) -> None:
    """Повторы и контактные действия для просмотренных объявлений."""
    # Навигация между объявлениями. В признаки не попадают сами значения item_id.
    it = cookie['item_id']
    mask = np.isfinite(it)
    ii = it[mask]
    ee = e[mask]
    tt = t[mask]
    if len(ii) > 1:
        pos = np.diff(tt) > 0
        same = ii[1:] == ii[:-1]
        out['item_adjacent_repeat'] = np.mean(same[pos]) if pos.any() else 0
        # Остаточный индекс повторяемости, а не вероятность; возможны отрицательные значения.
        out['item_return_nonadjacent'] = 1 - len(np.unique(ii)) / len(ii) - np.mean(same)
    else:
        out['item_adjacent_repeat'] = np.nan
        out['item_return_nonadjacent'] = np.nan
    iv = set(it[e == 1])
    iv.discard(np.nan)
    iv = {v for v in iv if np.isfinite(v)}
    for k in [2, 3, 4, 5, 6, 7]:
        vals = set(it[(e == k) & np.isfinite(it)])
        out[f'item_{EVENTS[k]}_viewed_share'] = len(vals & iv) / max(1, len(vals))
        out[f'item_view_with_{EVENTS[k]}_share'] = len(vals & iv) / max(1, len(iv))
    sizes = []
    types = []
    durations = []
    for v in np.unique(ii):
        ma = ii == v
        sizes.append(ma.sum())
        types.append(len(np.unique(ee[ma])))
        durations.append(tt[ma][-1] - tt[ma][0])
    stats(sizes, 'per_item_events', out, False)
    stats(types, 'per_item_types', out, False)
    stats(durations, 'per_item_duration', out, False)


def _add_query_navigation(cookie: dict, out: dict) -> None:
    """Листание внутри одного запроса, не между разными запросами."""
    q = cookie['search_query']
    pages = cookie['search_page']
    mask = pd.notna(q) & np.isfinite(pages)
    qs = q[mask]
    pg = pages[mask]
    deltas = []
    depths = []
    qcnt = []
    for v in np.unique(qs):
        p = pg[qs == v]
        d = np.diff(p)
        deltas.extend(d.tolist())
        depths.append(p.max())
        qcnt.append(len(p))
    stats(deltas, 'within_query_page_step', out, False)
    stats(depths, 'within_query_depth', out, False)
    stats(qcnt, 'within_query_searches', out, False)
    dd = np.array(deltas)
    for label, cond in [('up1', dd == 1), ('up', dd > 0), ('same', dd == 0), ('down', dd < 0)]:
        out['within_query_page_' + label] = np.mean(cond) if len(dd) else np.nan


def _add_pointer_geometry(cookie: dict, out: dict) -> None:
    """Разброс координат, направления и ближайшие соседние точки."""
    # Геометрия указателя. Масштаб 1920x1080 — нормировка диапазона этой выборки,
    # не правило классификации. При других viewport нужны новые проверки.
    x = cookie['pointer_x']
    y = cookie['pointer_y']
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    m = len(x)
    if m > 1:
        rx = np.ptp(x)
        ry = np.ptp(y)
        cov = np.cov(x, y, ddof=0)
        vx = cov[0, 0]
        vy = cov[1, 1]
        cxy = cov[0, 1]
        out['ptr_area'] = rx * ry
        out['ptr_area_corrected'] = rx * ry * ((m + 1) / (m - 1)) ** 2
        out['ptr_range_x_corrected'] = rx * (m + 1) / (m - 1)
        out['ptr_range_y_corrected'] = ry * (m + 1) / (m - 1)
        out['ptr_cov_det'] = np.linalg.det(cov)
        out['ptr_cov_xy'] = cxy
        out['ptr_cov_trace'] = vx + vy
        out['ptr_aspect'] = rx / (ry + 1)
        out['ptr_xy_corr'] = cxy / (np.sqrt(vx * vy) + 1e-06)
        eig = np.linalg.eigvalsh(cov)
        out['ptr_eig_ratio'] = (eig.min() + 1) / (eig.max() + 1)
        normdist = np.hypot((x - x.mean()) / 1920, (y - y.mean()) / 1080)
        stats(normdist, 'ptr_radius', out, False)
        dx = np.diff(x)
        dy = np.diff(y)
        disp = np.hypot(dx / 1920, dy / 1080)
        stats(disp, 'ptr_normalized_displacement', out, False)
        out['ptr_xsign_changes'] = np.mean(dx[1:] * dx[:-1] < 0) if len(dx) > 1 else np.nan
        out['ptr_ysign_changes'] = np.mean(dy[1:] * dy[:-1] < 0) if len(dy) > 1 else np.nan
        out['ptr_axis_aligned_share'] = np.mean((dx == 0) | (dy == 0))
        for axis, z in [('x', x), ('y', y)]:
            v = (z - z.mean()) / (z.std() + 1e-06)
            out['ptr_' + axis + '_skew'] = np.mean(v ** 3)
            out['ptr_' + axis + '_kurt'] = np.mean(v ** 4)
        dxp = x[:, None] - x
        dyp = y[:, None] - y
        dist = np.hypot(dxp / 1920, dyp / 1080)
        np.fill_diagonal(dist, np.inf)
        nearest = dist.min(axis=1)
        stats(nearest, 'ptr_nearest', out, False)
        out['ptr_small_cluster_share'] = np.mean(nearest < 0.025)


def build_extra(data_dir, meta, out_path=None):
    """Считает расширенный набор и выравнивает строки по метаданным."""
    start = time.time()
    ev = pd.read_csv(Path(data_dir) / 'events.csv.gz', parse_dates=['event_ts'])
    ev = ev.merge(meta[[
        'cookie_id',
        'window_start_ts',
        'window_end_ts',
    ]], on='cookie_id', validate='many_to_one')
    # Те же границы и правила удаления дублей, что в базовых признаках.
    ev = ev.loc[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)].drop_duplicates()
    ev = ev.sort_values([
        'cookie_id',
        'event_ts',
        'eid',
        'item_id',
        'search_page',
    ], kind='stable', na_position='last')
    ev['s'] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev['e'] = ev.event_name.map(EV_CODE)
    cols = [
        'cookie_id',
        's',
        'e',
        'item_id',
        'item_category',
        'item_location',
        'search_query',
        'search_page',
        'pointer_x',
        'pointer_y',
    ]
    ar = {c: ev[c].to_numpy() for c in cols}
    ids = ar['cookie_id']
    cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ev)]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for a, b in zip(cuts[:-1], cuts[1:]):
            cookie = {name: values[a:b] for name, values in ar.items()}
            t = cookie['s']
            e = cookie['e']
            dt = np.diff(t)
            f = {'cookie_id': ids[a]}
            _add_trimmed_gaps(dt, f)
            _add_gap_shape(dt, f)
            _add_session_rates(t, f)
            _add_rolling_counts(t, f)
            _add_transition_matrix(e, dt, f)
            _add_item_navigation(cookie, e, t, f)
            _add_query_navigation(cookie, f)
            _add_pointer_geometry(cookie, f)
            rows.append(f)
    X = pd.DataFrame(rows).set_index('cookie_id').reindex(meta.cookie_id)
    X.index = meta.index
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if out_path:
        X.to_pickle(out_path)
    print('Extra features:', X.shape, 'elapsed', round(time.time() - start, 2), flush=True)
    return X
