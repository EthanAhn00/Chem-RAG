"""Download pretrained GloVe vectors and convert them to a compact corpus-vocabulary .npz for a numpy-only dense encoder. This environment blocked HuggingFace transformer downloads, so GitHub releases of Wikipedia+Gigaword GloVe provide an alternative.
These are distributional vectors learned from corpus co-occurrence, unlike arbitrary hashing projections. They are not transformer sentence encoders (e.g. BAAI/bge); sentence vectors use IDF-weighted mean pooling, which must be disclosed in reporting.
Usage: python3 scripts/qdrant/qdrant_fetch_dense_model.py (glove-wiki-gigaword-300), --model glove-wiki-gigaword-100, or --no-prune to retain all vocabulary.
"""
import os
import sys
import gzip
import json
import time
import shutil
import hashlib
import argparse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from config.settings import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, 'embeddings_cache')
DERIVED_DIR = os.path.join(DATA_DIR, 'db_export', 'derived')

GITHUB_RELEASE = ('https://github.com/piskvorky/gensim-data/releases/download/'
                  '{model}/{model}.gz')
SUPPORTED = {
    'glove-wiki-gigaword-300': 300,
    'glove-wiki-gigaword-200': 200,
    'glove-wiki-gigaword-100': 100,
    'glove-wiki-gigaword-50': 50,
}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'  Reusing downloaded file: {dest} ({os.path.getsize(dest)/1e6:.0f}MB)')
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    print(f'  Downloading: {url}')
    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'chemrag/1.0'})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    print(f'  Complete: {os.path.getsize(dest)/1e6:.0f}MB, {time.time()-t0:.1f}s')
    return dest


def load_corpus_vocab():
    """Prune vectors using vocabulary from the fitted BM25 state."""
    path = os.path.join(DERIVED_DIR, 'bm25_state.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return set(json.load(f)['vocab'].keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='glove-wiki-gigaword-300', choices=sorted(SUPPORTED))
    ap.add_argument('--cache-dir', default=CACHE_DIR)
    ap.add_argument('--download-dir', default=None,
                    help='Directory for original .gz files (default: --cache-dir); use local disk for slow mounts')
    ap.add_argument('--no-prune', action='store_true', help='Keep all vocabulary instead of pruning to the corpus')
    args = ap.parse_args()

    dim = SUPPORTED[args.model]
    url = GITHUB_RELEASE.format(model=args.model)
    dl_dir = args.download_dir or args.cache_dir
    raw = os.path.join(dl_dir, f'{args.model}.gz')

    print(f'[1/3] Obtaining original vectors ({args.model}, {dim} dimensions)')
    download(url, raw)
    raw_sha = sha256_file(raw)

    print('[2/3] Pruning to corpus vocabulary')
    vocab_filter = None if args.no_prune else load_corpus_vocab()
    if vocab_filter is None:
        print('  (bm25_state.json missing; retaining full vocabulary. Index first to reduce size.)')
    else:
        print(f'  Pruning to corpus vocabulary of {len(vocab_filter)} terms')

    words, vecs = [], []
    with gzip.open(raw, 'rt', encoding='utf-8') as f:
        header = f.readline().split()
        total, file_dim = int(header[0]), int(header[1])
        if file_dim != dim:
            raise SystemExit(f'Dimension mismatch: file {file_dim} != expected {dim}')
        for line in f:
            parts = line.rstrip('\n').split(' ')
            w = parts[0]
            if vocab_filter is not None and w not in vocab_filter:
                continue
            words.append(w)
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    if not words:
        raise SystemExit('Pruning produced no vectors. Retry with --no-prune.')
    matrix = np.vstack(vecs)
    print(f'  Original {total}; retained {len(words)} -> {matrix.nbytes/1e6:.1f}MB')

    print('[3/3] Saving')
    os.makedirs(args.cache_dir, exist_ok=True)
    out_npz = os.path.join(args.cache_dir, f'{args.model}.npz')
    np.savez_compressed(out_npz, words=np.array(words, dtype=object), vectors=matrix)
    meta = {
        'model_id': args.model,
        'dim': dim,
        'source_url': url,
        'source_sha256': raw_sha,
        'source_vocab_size': total,
        'stored_vocab_size': len(words),
        'pruned_to_corpus_vocab': vocab_filter is not None,
        'kind': 'pretrained static word vectors (GloVe, Wikipedia+Gigaword)',
        'note': 'Not a transformer sentence encoder; sentence vectors use IDF-weighted mean pooling.',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    with open(os.path.join(args.cache_dir, f'{args.model}.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f'\nSaved: {out_npz}')


if __name__ == '__main__':
    main()
