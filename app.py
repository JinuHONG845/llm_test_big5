import streamlit as st
import json
import pandas as pd
import requests

# Streamlit secrets에서 API 키 가져오기
gpt40_api_key = st.secrets["GPT40_API_KEY"]
claude_api_key = st.secrets["CLAUDE_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# JSON 파일 로드
with open('persona.json', 'r') as f:
    personas = json.load(f)

with open('ipip.json', 'r') as f:
    ipip_questions = json.load(f)

with open('bif.json', 'r') as f:
    bif_questions = json.load(f)

# LLM 선택 버튼
st.title("Big5 성격 평가")
llm_choice = st.radio("LLM 선택", ("GPT-40", "Claude Sonnet 3.5", "Gemini Pro"))

# LLM에 따라 API 키 선택
if llm_choice == "GPT-40":
    api_key = gpt40_api_key
elif llm_choice == "Claude Sonnet 3.5":
    api_key = claude_api_key
else:
    api_key = gemini_api_key

# 점수 저장을 위한 데이터프레임 초기화
ipip_df = pd.DataFrame(columns=[f"Persona {i+1}" for i in range(len(personas))])
bif_df = pd.DataFrame(columns=[f"Persona {i+1}" for i in range(len(personas))])

# 각 persona에 대해 점수 계산 및 표출
for idx, persona in enumerate(personas):
    st.header(f"Persona {idx+1}: {persona['name']}")
    
    # IPIP 점수 표출
    st.subheader("IPIP 점수")
    ipip_scores = {q['question']: st.slider(q['question'], 1, 5) for q in ipip_questions}
    ipip_df[f"Persona {idx+1}"] = list(ipip_scores.values())
    
    # BIF 점수 표출
    st.subheader("BIF 점수")
    bif_scores = {q['question']: st.slider(q['question'], 1, 5) for q in bif_questions}
    bif_df[f"Persona {idx+1}"] = list(bif_scores.values())

# 점수 테이블 표출
st.subheader("IPIP 점수 테이블")
st.dataframe(ipip_df)

st.subheader("BIF 점수 테이블")
st.dataframe(bif_df)

# Gemini API 호출 함수
def call_gemini_api(api_key, persona, ipip_scores, bif_scores):
    url = "https://api.gemini.com/v1/your-endpoint"  # 실제 엔