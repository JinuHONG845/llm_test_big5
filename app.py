import streamlit as st

# 기본 타이틀 설정
st.title("Big5 성격 평가")

# 간단한 라디오 버튼
llm_choice = st.radio("LLM 선택", ("GPT-40", "Claude Sonnet 3.5", "Gemini Pro"))
st.write(f"선택된 LLM: {llm_choice}")

# 간단한 슬라이더
test_score = st.slider("테스트 점수", 1, 5, 3)
st.write(f"선택된 점수: {test_score}")