# 1. 베이스가 될 공식 Ollama 이미지를 가져옵니다.
FROM ollama/ollama

# 2. 빌드 확인용 메시지를 출력하고, llama3.2:3b 모델을 다운로드합니다.
RUN echo "--- CHECKPOINT: 지금 llama3.2:3b를 설치하는 Dockerfile을 읽고 있습니다 ---" && \
    ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b