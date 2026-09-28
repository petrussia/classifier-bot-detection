"""Проверки метрики, признаков и формата ответа: python tests.py."""
from __future__ import annotations
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from features import build_features
from features_extra import build_extra
from metric import precision_at_recall
from solution import validate_submission


class MetricTests(unittest.TestCase):
    def test_official_examples_and_ties(self):
        self.assertEqual(precision_at_recall([1, 0, 1, 1, 1], [5, 4, 3, 2, 1]), .8)
        self.assertEqual(precision_at_recall([1, 1, 0, 0], [.5]*4), .5)
        self.assertEqual(precision_at_recall([0, 0, 1, 1], [.5]*4), .5)

    def test_against_independent_brute_force(self):
        rng = np.random.default_rng(2026)
        for _ in range(200):
            n = int(rng.integers(10, 150))
            y = rng.binomial(1, .15, size=n)
            y[0] = 1
            p = rng.integers(0, 11, size=n) / 10  # намеренно много одинаковых score
            valid = []
            for t in np.unique(p):
                flagged = p >= t
                if y[flagged].sum() / y.sum() >= .7:
                    valid.append(y[flagged].mean())
            self.assertAlmostEqual(precision_at_recall(y, p), max(valid), places=14)
            perm = rng.permutation(n)
            self.assertEqual(precision_at_recall(y, p), precision_at_recall(y[perm], p[perm]))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.start = pd.Timestamp('2026-01-02')
        meta = pd.DataFrame({
            'cookie_id': ['toy_a', 'toy_b', 'toy_c'],
            'cookie_created_at': [self.start-pd.Timedelta(days=3)]*3,
            'window_start_ts': [self.start]*3,
            'window_end_ts': [self.start+pd.Timedelta(days=1)]*3,
        })
        meta.iloc[:2].assign(target=[0, 1]).to_csv(self.root/'train.csv', index=False)
        meta.iloc[2:].to_csv(self.root/'test.csv', index=False)
        self.meta = meta
        def event(cookie, second, code, item=None):
            return {'cookie_id': cookie, 'event_ts': self.start+pd.Timedelta(seconds=second),
                    'eid': code, 'event_name': 'item_view' if code == 200 else 'search_results_view',
                    'platform': 'WEB',
                    'user_agent': 'Mozilla/5.0 Windows Chrome/120.0.0.0',
                    'item_id': item, 'item_category': 'example', 'item_location': 'example_city',
                    'seller_type': 'private' if item else None,
                    'search_query': 'query' if code == 100 else None,
                    'search_page': 1 if code == 100 else None,
                    'pointer_x': 100+second % 30, 'pointer_y': 200+second % 20}
        self.events = pd.DataFrame([
            event('toy_a', 0, 100), event('toy_a', 10, 200, 10),
            event('toy_a', 10, 200, 10),  # полный дубликат
            event('toy_b', 120, 200, 20), event('toy_c', 300, 200, 30),
            event('toy_a', -1, 200, 50),  # до окна
            event('toy_a', 86400, 200, 60),  # на правой границе
            event('toy_a', 86401, 200, 70),  # после окна
        ])
        self.events.to_csv(self.root/'events.csv.gz', index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def extract(self):
        with contextlib.redirect_stdout(io.StringIO()):
            meta, X, audit = build_features(self.root)
            Z = build_extra(self.root, meta)
        return meta, X, Z, audit

    def test_boundaries_duplicates_and_grain(self):
        meta, X, Z, audit = self.extract()
        np.testing.assert_array_equal(X.n_events.to_numpy(), [2, 1, 1])
        self.assertEqual(audit['events_before_window'], 1)
        self.assertEqual(audit['events_at_or_after_end'], 2)
        self.assertEqual(audit['duplicates_within_window'], 1)
        self.assertEqual(len(X), len(meta))
        forbidden = {'target', 'cookie_id', 'event_ts', 'cookie_created_at',
                     'window_start_ts', 'window_end_ts', 'item_id', 'user_agent'}
        self.assertFalse(forbidden & set(X.columns))
        self.assertFalse(forbidden & set(Z.columns))

    def test_invariance_to_future_events_and_labels(self):
        _, X1, Z1, _ = self.extract()
        # Изменение меток и будущих событий не должно менять признаки.
        tr = pd.read_csv(self.root/'train.csv')
        tr['target'] = 1-tr.target
        tr.to_csv(self.root/'train.csv', index=False)
        extra = self.events.iloc[[-1]].copy()
        extra['pointer_x'] = 10**8
        extra['item_id'] = 123456789
        pd.concat([self.events, extra], ignore_index=True).to_csv(self.root/'events.csv.gz', index=False)
        _, X2, Z2, _ = self.extract()
        pd.testing.assert_frame_equal(X1, X2, check_like=True)
        pd.testing.assert_frame_equal(Z1, Z2, check_like=True)

    def test_cookie_identifiers_do_not_affect_features(self):
        _, X1, Z1, _ = self.extract()
        mapping = {'toy_a': 'new_z', 'toy_b': 'new_y', 'toy_c': 'new_x'}
        for filename in ['train.csv', 'test.csv']:
            m = pd.read_csv(self.root/filename)
            m['cookie_id'] = m.cookie_id.map(mapping)
            m.to_csv(self.root/filename, index=False)
        ev = self.events.copy()
        ev['cookie_id'] = ev.cookie_id.map(mapping)
        ev.to_csv(self.root/'events.csv.gz', index=False)
        _, X2, Z2, _ = self.extract()
        pd.testing.assert_frame_equal(X1, X2, check_like=True)
        pd.testing.assert_frame_equal(Z1, Z2, check_like=True)


class SubmissionTests(unittest.TestCase):
    def test_valid_format(self):
        test = pd.DataFrame({'cookie_id': ['a', 'b']})
        validate_submission(pd.DataFrame({'cookie_id': ['a', 'b'], 'score': [.1, .9]}), test)

    def test_reject_invalid_rows_or_scores(self):
        test = pd.DataFrame({'cookie_id': ['a', 'b']})
        bad = [(['a', 'a'], [.1, .9]), (['a', 'b'], [np.nan, .9]),
               (['a', 'b'], [-.1, .9]), (['b', 'a'], [.1, .9])]
        for ids, p in bad:
            with self.assertRaises(ValueError):
                validate_submission(pd.DataFrame({'cookie_id': ids, 'score': p}), test)


if __name__ == '__main__':
    unittest.main(verbosity=2)
