# backend.py

import os
import re
import subprocess
import pathlib
import requests
from dotenv import load_dotenv
from Embedding import watsonx_embedding, vectorstore

# 1. 환경 변수 로드
load_dotenv()
WATSONX_API = os.environ['API_KEY']
PROJECT_ID = os.environ['PROJECT_ID']
IBM_URL = os.environ['IBM_CLOUD_URL']
WATSONX_ENDPOINT = 'https://us-south.ml.cloud.ibm.com/ml/v1/deployments/47f04daa-ce50-4317-8712-bef7dc271031/text/generation?version=2021-05-01'


# 2. 사전 정의된 스미싱 메시지/URL 목록 불러오기
with open("./smishing_URL.csv", "r", encoding="utf-8") as f:
    known_urls = set(line.strip() for line in f if line.strip())
with open("./labeled_smishing_messages.csv", "r", encoding="utf-8") as f:
    known_messages = set(line.strip() for line in f if line.strip())


# 3. 사용자 입력 메시지에서 URL 추출하는 함수 (1단계에서 단순 비교)
def extract_urls(text: str):
    url_pattern = r"""(?i)\b(https?://|www\\.)?[a-z0-9.-]+\\.[a-z]{2,}(/[\\w./?%&=:#@!~+-]*)?"""
    return re.findall(url_pattern, text)


# 4. Vector DB 존재 여부 확인하는 함수
VECTOR_DB_DIR = "./chroma_store"
def ensure_vector_db():
    db_ready = pathlib.Path(VECTOR_DB_DIR).exists() and len(os.listdir(VECTOR_DB_DIR)) > 0
    if not db_ready:
        print("❗️벡터 DB가 존재하지 않아 Embedding.py를 실행합니다...")
        subprocess.run(["python", "Embedding.py"], check=True)
    else:
        print("✅ 기존 벡터 DB가 확인되어 바로 로드합니다.")


# 5. Watsonx Prompt Lab Endpoint 호출하는 함수
def call_watsonx_endpoint(user_input: str, similar_cases: str) -> str:
    # (1) IBM Cloud IAM 인증 토큰 발급
    token_response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "apikey": WATSONX_API,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
        }
    )
    mltoken = token_response.json().get("access_token")
    if not mltoken:
        raise Exception("❌ IBM Cloud 토큰 발급 실패")

    # (2) 요청 헤더
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mltoken}",
        "Accept": "text/event-stream"
    }

    # (3) Watsonx Prompt Lab 호출용 페이로드
    payload = {
        "parameters": {
            "prompt_variables": {
                "user_input": user_input,
                "similar_cases": similar_cases
            }
        }
    }

    # (4) Watsonx API 요청
    response = requests.post(
        WATSONX_ENDPOINT,
        headers=headers,
        json=payload,
        stream=False
    )

    # (5) 응답 체크
    try:
        response_json = response.json()
        return response_json["results"][0]["generated_text"].strip()
    except Exception as e:
        raise Exception(f"❌ Watsonx 응답 파싱 실패: {e}\n📭 응답 원문: {response.text}")


# 6. 응답 파싱 함수
def parse_ai_response(response_text: str) -> dict:
    try:
        # 1. label 추출: '1. 최종 판단:' ~ '2. 판단 근거' 이전까지
        label_match = re.search(r"1\.\s*최종 판단[:：]?\s*(.*?)\n\s*2\.", response_text, re.DOTALL)
        label = label_match.group(1).strip() if label_match else "분석 실패"

        # 2. reason 추출: '2. 판단 근거:' ~ '3. 위험도' 이전까지
        reason_match = re.search(r"2\.\s*판단 근거[:：]?\s*((?:.|\n)*?)\n\s*3\.", response_text)
        reason = reason_match.group(1).strip() if reason_match else "판단 근거를 찾을 수 없습니다."

        # 3. confidence 추출: '3. 위험도:' ~ '%'까지 포함
        confidence_match = re.search(r"3\.\s*위험도[:：]?\s*([^\n%]+%)", response_text)
        confidence = confidence_match.group(1).strip() if confidence_match else "0%"

        return {
            "label": label,
            "confidence": confidence,
            "reason": reason
        }
    except Exception as e:
        return {
            "label": "판단 실패",
            "confidence": "0%",
            "reason": f"❌ 응답 파싱 오류: {e}"
        }

# 7. 최종 판단 함수
def classify_message(user_input: str) -> dict:
    cleaned_input = user_input.strip()

    # (1) 1차 사전 필터링
    urls_in_message = extract_urls(cleaned_input)
    if cleaned_input in known_messages:
        return {
            "label": "스미싱",
            "confidence": 1.0,
            "reason": "사전 등록된 스미싱 메시지와 일치합니다."
        }
    if any(url in known_urls for url in urls_in_message):
        return {
            "label": "스미싱",
            "confidence": 1.0,
            "reason": "사전 등록된 스미싱 URL이 포함되어 있습니다."
        }


    # (2) 임베딩 / Vector DB 검색
    ensure_vector_db() # VectorDB가 이미 구축되어 있는지 확인하는 함수
    try:
        # 1. 입력 메시지를 임베딩
        query_embedding = watsonx_embedding.embed_query(cleaned_input)

        # 2. 유사 메시지 검색 (예: 상위 3개)
        docs = vectorstore.similarity_search_by_vector(query_embedding, k=3)

        # 3. 유사 메시지를 문자열로 정리
        retrieved_texts = "\n".join([f"- {doc.page_content}" for doc in docs])

    except Exception as e:
        return {
            "label": "판단 실패",
            "confidence": 0.0,
            "reason": f"Vector DB 검색 중 오류 발생: {e}"
        }


    # (3) 2차 AI 판단 (왓슨 호출)
    try:
        ai_response = call_watsonx_endpoint(user_input=cleaned_input, similar_cases=retrieved_texts)
        print("📭 Watsonx 응답 텍스트:")
        parsed_result = parse_ai_response(ai_response)

        return parsed_result
    
    except Exception as e:
        return {
            "label": "판단 실패",
            "confidence": 0.0,
            "reason": f"AI 모델 호출 중 오류 발생: {e}"
        }


# 8. 보호자에게 알림 보내는 함수

# 9. 대응 가이드 안내하는 함수