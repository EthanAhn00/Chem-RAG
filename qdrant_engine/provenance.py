"""Track input CSV content hashes to generate payload source_version."""
import hashlib
import os


def file_digest(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_version(paths):
    """Combine multiple source-file content hashes into one short version string. Changes reveal reindexing needs from payload provenance."""
    combined = hashlib.sha256()
    detail = {}
    for p in sorted(paths):
        if not os.path.exists(p):
            detail[os.path.basename(p)] = 'MISSING'
            combined.update(f'{os.path.basename(p)}:MISSING'.encode())
            continue
        d = file_digest(p)
        detail[os.path.basename(p)] = d[:12]
        combined.update(f'{os.path.basename(p)}:{d}'.encode())
    return combined.hexdigest()[:16], detail


def runtime_provenance(settings=None, command=None):
    """Execution-environment metadata for reproducibility in result files."""
    import sys
    import time
    import platform
    import subprocess
    from importlib.metadata import version, PackageNotFoundError

    def pkg(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    git = {}
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git['commit'] = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True, timeout=15).stdout.strip() or None
        st = subprocess.run(['git', '-C', root, 'status', '--porcelain'],
                            capture_output=True, text=True, timeout=15).stdout.strip()
        git['working_tree'] = 'clean' if not st else f'dirty ({len(st.splitlines())} entries)'
    except Exception as exc:
        git = {'error': f'{type(exc).__name__}: {exc}'}

    prov = {
        'run_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'command': command,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': {n: pkg(n) for n in ('qdrant-client', 'numpy', 'rdkit')},
        'git': git,
    }
    if settings is not None:
        prov['qdrant'] = {
            'mode': 'server' if settings.url else 'local(embedded)',
            'url_or_path': settings.url or settings.local_path,
            'patents_collection': settings.patents_collection,
            'structures_collection': settings.structures_collection,
            'dense_backend': settings.dense_backend,
            'dense_model': settings.dense_model,
            'bm25': {'k1': settings.bm25_k1, 'b': settings.bm25_b},
        }
    return prov
