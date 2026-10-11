"""텍스트 -> 벡터 인코더 패키지 (dense / sparse).

Qdrant 적재·검색 코드가 특정 임베딩 라이브러리에 직접 의존하지 않도록
base.Encoder 인터페이스와 registry를 통해 백엔드를 갈아끼울 수 있게 한다.
"""
