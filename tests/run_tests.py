"""테스트 러너 - unit / integration / full-index 를 구분해 실행하고 집계한다.

  unit         : DB 없이 실행
  integration  : 임시 Qdrant 컬렉션을 직접 만들어 검색
  full-index   : 실제 4,123 / 6,928 포인트 DB 필요 (scripts/qdrant/qdrant_verify_index.py)

skip 된 테스트는 PASS로 세지 않는다. 실패나 에러가 있으면 종료 코드 1.

    python3 tests/run_tests.py
    python3 tests/run_tests.py --skip-full-index
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
            # 전제 조건 미충족은 SKIP, 그 외에는 FAIL
            if isinstance(exc, FileNotFoundError):
                skipped += 1
                print(f'  [SKIP] {name} - 전제 데이터 없음: {exc}')
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
        print('\n===== full-index verification =====\n  [SKIP] --skip-full-index 지정됨')
    else:
        print('\n===== full-index verification (scripts/qdrant/qdrant_verify_index.py) =====')
        proc = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'qdrant', 'qdrant_verify_index.py')],
                              capture_output=True, text=True)
        tail = [l for l in proc.stdout.splitlines() if l.strip()][-3:]
        for l in tail:
            print('  ' + l)
        print(f'  종료 코드: {proc.returncode}')
        if proc.returncode == 0:
            full['passed'] = 1
            full['details'].append({'test': 'qdrant_verify_index', 'status': 'PASS'})
        else:
            full['failed'] = 1
            full['details'].append({'test': 'qdrant_verify_index', 'status': 'FAIL',
                                    'reason': (proc.stderr or proc.stdout)[-400:]})
    results.append(full)

    print('\n================ 요약 ================')
    tp = tf = ts = 0
    for r in results:
        print(f"  {r['suite']:26} pass={r['passed']:3} fail={r['failed']:3} skip={r['skipped']:3}")
        tp += r['passed']; tf += r['failed']; ts += r['skipped']
    print(f"  {'TOTAL':26} pass={tp:3} fail={tf:3} skip={ts:3}")
    print('  (skip은 PASS에 포함되지 않음)')
    return 1 if tf else 0


if __name__ == '__main__':
    sys.exit(main())
