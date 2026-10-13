"""Qdrant patent/chemical-structure vector search.
config.py: environment/JSON settings; secrets only via environment variables.
client.py: client factory, collections, aliases.
schemas.py: schemas, payload indexes, UUID5 point IDs.
filters.py: time/family/self-exclusion filters.
patent_retriever.py: BM25/dense/hybrid retrieval.
structure_repository.py: structure lookup and RDKit exact Tanimoto.
hybrid_fusion.py: RRF/DBSF fusion, structure reranking, weight renormalization.
"""
