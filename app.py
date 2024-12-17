import streamlit as st
import json
import pandas as pd
import requests

# Streamlit secrets에서 API 키 가져오기
gpt40_api_key = st.secrets.get("GPT40_API_KEY")
claude_api_key = st.secrets.get("CLAUDE_API_KEY")
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

# JSON 파일 로드
def load_json(file_name):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"{file_name} 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        st.error(f"{file_name} 파일을 파싱하는 데 실패했습니다.")
        return []

personas = load_json('persona.json')
ipip_questions = load_json('ipip.json')
bif_questions = load_json('bif.json')

# LLM 선택 버튼
st.title("Big5 성격 평가")
llm_choice = st.radio("LLM 선택", ("GPT-40", "Claude Sonnet 3.5", "Gemini Pro"))

# LLM에 따라 API 키 선택
api_key = {
    "GPT-40": gpt40_api_key,
    "Claude Sonnet 3.5": claude_api_key,
    "Gemini Pro": gemini_api_key
}.get(llm_choice)

if not api_key:
    st.error("API 키가 설정되지 않았습니다. secrets.toml 파일을 확인하세요.")

# 점수 저장을 위한 데이터프레임 초기화
ipip_df = pd.DataFrame(columns=[f"Persona {i+1}" for i in range(len(personas))])
bif_df = pd.DataFrame(columns=[f"Persona {i+1}" for i in range(len(personas))])

# 각 persona에 대해 점수 계산 및 표출
for idx, persona in enumerate(personas):
    st.header(f"Persona {idx+1}: {persona.get('name', 'Unknown')}")
    
    # IPIP 점수 표출
    ipip_scores = {q['question']: st.slider(q['question'], 1, 5) for q in ipip_questions}
    ipip_df[f"Persona {idx+1}"] = list(ipip_scores.values())
    
    # BIF 점수 표출
    bif_scores = {q['question']: st.slider(q['question'], 1, 5) for q in bif_questions}
    bif_df[f"Persona {idx+1}"] = list(bif_scores.values())

# 점수 테이블 표출
st.subheader("IPIP 점수 테이블")
st.dataframe(ipip_df)

st.subheader("BIF 점수 테이블")
st.dataframe(bif_df)

# Gemini API 호출 함수
def call_gemini_api(api_key, persona, ipip_scores, bif_scores):
    url = "https://api.gemini.com/v1/your-endpoint"  # 실제 엔드포인트로 변경하세요
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "persona": persona,
        "ipip_scores": ipip_scores,
        "bif_scores": bif_scores
    }
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# 결과 확인 버튼
if st.button("결과 확인"):
    for idx, persona in enumerate(personas):
        result = call_gemini_api(api_key, persona, ipip_df[f"Persona {idx+1}"].to_dict(), bif_df[f"Persona {idx+1}"].to_dict())
        if result:
            st.write(f"Persona {idx+1} 결과: {result}")