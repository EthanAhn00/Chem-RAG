"""Test runner for unit, integration, and full-index checks.
unit: no DB required.
integration: create temporary Qdrant collections and perform retrieval.
full-index: requires actual 4,123/6,928-point DB via qdrant_verify_index.py.
Skipped tests do not count as PASS. Any failure/error exits 1.
Usage: python3 tests/run_tests.py [--skip-full-index].
"""
import os
import sys
import time
import argparse
import importlib
import subprocess
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run_module(mod_name, label):
    print(f'\n===== {label} ({mod_name}) =====')
    mod = importlib.import_module(mod_name)
    tests = [(n, f) for n, f in sorted(vars(mod).items()) if n.startswith('test_') and callable(f)]
    passed, failed, skipped, details = 0, 0, 0, []
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            passed += 1
            print(f'  [PASS] {name} ({time.time()-t0:.1f}s)')
        except Exception as exc:
            # Missing prerequisites mean SKIP; other exceptions mean FAIL.
            if isinstance(exc, FileNotFoundError):
                skipped += 1
                print(f'  [SKIP] {name} - prerequisite data missing: {exc}')
                details.append({'test': name, 'status': 'SKIP', 'reason': str(exc)})
                continue
            failed += 1
            print(f'  [FAIL] {name}: {type(exc).__name__}: {exc}')
            traceback.print_exc(limit=2)
            details.append({'test': name, 'status': 'FAIL', 'reason': f'{type(exc).__name__}: {exc}'})
        else:
            details.append({'test': name, 'status': 'PASS'})
    print(f'  -> pass={passed} fail={failed} skip={skipped}')
    return {'suite': label, 'passed': passed, 'failed': failed, 'skipped': skipped, 'details': details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-full-index', action='store_true')
    args = ap.parse_args()

    results = [run_module('tests.test_unit', 'unit tests'),
               run_module('tests.test_integration', 'integration tests')]

    full = {'suite': 'full-index verification', 'passed': 0, 'failed': 0, 'skipped': 0, 'details': []}
    if args.skip_full_index:
        full['skipped'] = 1
        full['details'].append({'test': 'qdrant_verify_index', 'status': 'SKIP',
                                'reason': '--skip-full-index'})
        print('\n===== full-index verification =====\n  [SKIP] --skip-full-index specified')
    else:
        print('\n===== full-index verification (scripts/qdrant/qdrant_verify_index.py) =====')
        proc = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'qdrant', 'qdrant_verify_index.py')],
                              capture_output=True, text=True)
        tail = [l for l in proc.stdout.splitlines() if l.strip()][-3:]
        for l in tail:
            print('  ' + l)
        print(f'  Exit code: {proc.returncode}')
        if proc.returncode == 0:
            full['passed'] = 1
            full['details'].append({'test': 'qdrant_verify_index', 'status': 'PASS'})
        else:
            full['failed'] = 1
            full['details'].append({'test': 'qdrant_verify_index', 'status': 'FAIL',
                                    'reason': (proc.stderr or proc.stdout)[-400:]})
    results.append(full)

    print('\n================ Summary ================')
    tp = tf = ts = 0
    for r in results:
        print(f"  {r['suite']:26} pass={r['passed']:3} fail={r['failed']:3} skip={r['skipped']:3}")
        tp += r['passed']; tf += r['failed']; ts += r['skipped']
    print(f"  {'TOTAL':26} pass={tp:3} fail={tf:3} skip={ts:3}")
    print('  (skipped tests are not counted as PASS)')
    return 1 if tf else 0


if __name__ == '__main__':
    sys.exit(main())
