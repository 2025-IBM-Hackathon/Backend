# Embedding.py

import os
import pandas as pd
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter



# 1. .env 환경변수 불러오기

load_dotenv()

WATSONX_API = os.environ['API_KEY']
PROJECT_ID = os.environ['PROJECT_ID']
IBM_URL = os.environ['IBM_CLOUD_URL']



# 2. 데이터 로드 (정상 + 스미싱)

df1 = pd.read_csv("labeled_normal_messages.csv")    # 정상
df2 = pd.read_csv("labeled_smishing_messages.csv")  # 스미싱
df = pd.concat([df1, df2], ignore_index=True)


# 3. 문서 준비
texts = df["message"].tolist()
metadatas = [{"label": label} for label in df["label"].tolist()]


# 4. 텍스트 청크 분할

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80, # 평균적인 문자메시지 길이 고려
    chunk_overlap=10,
    length_function=len
)

docs = text_splitter.create_documents(texts, metadatas=metadatas)


# 5. 임베딩 모델 설정

# MODEL_ID = 'ibm/slate-125m-english-rtrvr'
MODEL_ID = 'ibm/granite-embedding-278m-multilingual' # English -> Multilingual
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3, # 입력 텍스트가 모델 최대 토큰 수를 초과할 경우 가장 뒤쪽을 자른다는 의미
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id=MODEL_ID,
    url=IBM_URL,
    project_id=PROJECT_ID,
    params=embed_params,
    apikey=WATSONX_API
)

# 6. ChromaDB 설정 및 저장

persist_directory = "./chroma_store" # Local에 저장할 경로

vectorstore = Chroma(
    collection_name="sms_messages",
    embedding_function=watsonx_embedding,
    persist_directory=persist_directory
)
vectorstore.add_documents(docs)
# vectorstore.persist()

print("✅ 임베딩 및 Vector DB 저장 완료!")
