"""실제 사전학습 semantic 임베딩 벡터를 내려받아 프로젝트용 압축 모델로 변환한다.

이 환경에서는 HuggingFace(huggingface.co)가 차단되어 fastembed/sentence-transformers로
transformer 문장 인코더를 받을 수 없다. 대신 GitHub 릴리스에서 배포되는 GloVe
사전학습 단어 벡터(Wikipedia+Gigaword)를 받아, 코퍼스 어휘만 남긴 .npz로 변환해
numpy만으로 동작하는 semantic dense 인코더를 만든다.

  - 사전학습된 분포 의미(distributional semantics) 벡터다. hashing 백엔드처럼
    문자열을 임의 차원에 투영하는 것이 아니라, 대규모 말뭉치의 단어 공기(co-occurrence)
    통계로 학습된 실제 의미 벡터다.
  - 단, transformer 문장 인코더(BAAI/bge 등)는 아니다. 문장 표현은 IDF 가중 평균
    풀링으로 만든다. 이 차이는 결과 보고서에 반드시 함께 적는다.

사용법:
    python3 scripts/qdrant/qdrant_fetch_dense_model.py                     # glove-wiki-gigaword-300
    python3 scripts/qdrant/qdrant_fetch_dense_model.py --model glove-wiki-gigaword-100
    python3 scripts/qdrant/qdrant_fetch_dense_model.py --no-prune          # 전체 어휘 유지
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
        print(f'  이미 받아둔 파일 사용: {dest} ({os.path.getsize(dest)/1e6:.0f}MB)')
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    print(f'  다운로드: {url}')
    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'chemrag/1.0'})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    print(f'  완료: {os.path.getsize(dest)/1e6:.0f}MB, {time.time()-t0:.1f}s')
    return dest


def load_corpus_vocab():
    """BM25 학습 결과(코퍼스 어휘)를 재사용해 벡터를 프루닝한다."""
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
                    help='원본 .gz를 둘 경로(기본: --cache-dir). 마운트가 느리면 로컬 디스크 지정')
    ap.add_argument('--no-prune', action='store_true', help='코퍼스 어휘로 자르지 않고 전부 저장')
    args = ap.parse_args()

    dim = SUPPORTED[args.model]
    url = GITHUB_RELEASE.format(model=args.model)
    dl_dir = args.download_dir or args.cache_dir
    raw = os.path.join(dl_dir, f'{args.model}.gz')

    print(f'[1/3] 원본 벡터 확보 ({args.model}, {dim}차원)')
    download(url, raw)
    raw_sha = sha256_file(raw)

    print('[2/3] 코퍼스 어휘로 프루닝')
    vocab_filter = None if args.no_prune else load_corpus_vocab()
    if vocab_filter is None:
        print('  (bm25_state.json 없음 - 전체 어휘를 저장합니다. 먼저 색인을 돌리면 크기가 줄어듭니다)')
    else:
        print(f'  코퍼스 어휘 {len(vocab_filter)}개 기준으로 프루닝')

    words, vecs = [], []
    with gzip.open(raw, 'rt', encoding='utf-8') as f:
        header = f.readline().split()
        total, file_dim = int(header[0]), int(header[1])
        if file_dim != dim:
            raise SystemExit(f'차원 불일치: 파일 {file_dim} != 예상 {dim}')
        for line in f:
            parts = line.rstrip('\n').split(' ')
            w = parts[0]
            if vocab_filter is not None and w not in vocab_filter:
                continue
            words.append(w)
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    if not words:
        raise SystemExit('프루닝 결과가 비었습니다. --no-prune 으로 다시 시도하세요.')
    matrix = np.vstack(vecs)
    print(f'  원본 {total}개 중 {len(words)}개 유지 -> {matrix.nbytes/1e6:.1f}MB')

    print('[3/3] 저장')
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
        'note': 'transformer 문장 인코더가 아님. 문장 벡터는 IDF 가중 평균 풀링으로 생성.',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    with open(os.path.join(args.cache_dir, f'{args.model}.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f'\n저장됨: {out_npz}')


if __name__ == '__main__':
    main()
