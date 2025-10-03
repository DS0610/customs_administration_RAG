import redis
import requests
import numpy as np
from redis.commands.search.query import Query

# 1. Redis 연결
r = redis.Redis(host="localhost", port=6379, decode_responses=False)

# 2. Ollama 임베딩 함수 (임베딩 전용 모델 사용)
def get_embedding(text: str):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": "nomic-embed-text", "prompt": text}  # ✅ 임베딩 모델
    res = requests.post(url, json=payload)
    data = res.json()
    if "embedding" not in data:
        raise ValueError(f"Ollama 임베딩 오류 응답: {data}")
    return data["embedding"]

# 3. Ollama 생성 함수 (답변 생성)
def generate_answer(prompt: str, context: str = ""):
    url = "http://localhost:11434/api/generate"
    full_prompt = f"질문: {prompt}\n\n참고 문서: {context}\n\n답변:"
    payload = {"model": "llama3.2:3b", "prompt": full_prompt, "stream": False}
    res = requests.post(url, json=payload)
    data = res.json()
    return data.get("response", "⚠️ 모델 응답 오류")

# 4. 임베딩 차원 자동 확인
test_emb = get_embedding("차원 확인 테스트")
dim = len(test_emb)
print(f"✅ 임베딩 차원: {dim}")

# 5. 인덱스 생성 (기존 있으면 삭제)
index_name = "qa_index"
try:
    r.ft(index_name).dropindex(delete_documents=True)
except:
    pass

r.ft(index_name).create_index(
    fields=[
        redis.commands.search.field.VectorField("embedding", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": dim,
            "DISTANCE_METRIC": "COSINE"
        }),
        redis.commands.search.field.TextField("value")
    ]
)

# 6. 예시 데이터 (Key=질문, Value=답변)
qa_data = {
    "대한민국의 수도는 어디야?": "대한민국의 수도는 서울입니다.",
    "미국의 수도는 어디야?": "미국의 수도는 워싱턴 D.C.입니다.",
    "일본의 수도는 어디야?": "일본의 수도는 도쿄입니다.",
    "중국의 수도는 어디야?": "중국의 수도는 베이징입니다.",
    "프랑스의 수도는 어디야?": "프랑스의 수도는 파리입니다."
}

# 7. Redis에 데이터 적재
for i, (q, a) in enumerate(qa_data.items()):
    emb = get_embedding(q)
    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
    r.hset(f"doc:{i}", mapping={
        "embedding": emb_bytes,
        "value": a
    })

print("✅ Redis에 데이터 적재 완료")

# 8. 질의 → Redis 검색 → 답변 반환
query = "한국의 수도는 어디입니까?"
query_emb = get_embedding(query)
query_emb_bytes = np.array(query_emb, dtype=np.float32).tobytes()

q = Query('*=>[KNN 1 @embedding $vec]') \
    .return_fields("value", "__score__") \
    .dialect(2)

res = r.ft(index_name).search(
    q,
    query_params={"vec": query_emb_bytes}
)

if res.docs:
    cached_answer = res.docs[0]["value"]
    print("🔍 Redis 검색 결과:", cached_answer)
else:
    print("⚠️ Redis에서 검색 결과 없음")
    cached_answer = None

# 9. 캐시 미스 → Ollama 생성 모델 호출
if cached_answer:
    final_answer = cached_answer
else:
    final_answer = generate_answer(query)

print("🤖 최종 답변:", final_answer)
