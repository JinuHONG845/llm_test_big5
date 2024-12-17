import streamlit as st
import pandas as pd
import json

# 기본 타이틀 설정
st.title("Big5 성격 평가")

try:
    # JSON 파일 로드
    with open('persona.json', 'r') as f:
        personas = json.load(f)
    
    with open('ipip.json', 'r') as f:
        ipip_questions = json.load(f)
        
    with open('bif.json', 'r') as f:
        bif_questions = json.load(f)

    # LLM 선택 버튼
    llm_choice = st.radio("LLM 선택", ("GPT-40", "Claude Sonnet 3.5", "Gemini Pro"))
    st.write(f"선택된 LLM: {llm_choice}")

    # API 키 설정
    api_key = st.secrets.get(f"{llm_choice.replace(' ', '_').upper()}_API_KEY")
    if not api_key:
        st.warning("API 키가 설정되지 않았습니다.")

    # 점수 저장을 위한 데이터프레임 초기화
    ipip_df = pd.DataFrame(columns=[f"Persona {i+1}" for i in range(len(personas))])
    bif_df = pd.DataFrame(columns=[f"Persona {i+1}" for i in range(len(personas))])

    # 각 persona에 대해 점수 계산 및 표출
    for idx, persona in enumerate(personas):
        st.header(f"Persona {idx+1}: {persona.get('name', 'Unknown')}")
        
        # IPIP 점수 표출
        st.subheader("IPIP 점수")
        ipip_scores = {}
        for q in ipip_questions:
            score = st.slider(
                q['question'],
                min_value=1,
                max_value=5,
                value=3,
                help="1=Very inaccurate, 2=Moderately inaccurate, 3=Neither, 4=Moderately accurate, 5=Very accurate"
            )
            ipip_scores[q['question']] = score
        ipip_df[f"Persona {idx+1}"] = list(ipip_scores.values())
        
        # BIF 점수 표출
        st.subheader("BIF 점수")
        bif_scores = {}
        for q in bif_questions:
            score = st.slider(
                q['question'],
                min_value=1,
                max_value=5,
                value=3,
                help="1=Disagree Strongly, 2=Disagree a little, 3=Neither agree nor disagree, 4=Agree a little, 5=Agree strongly"
            )
            bif_scores[q['question']] = score
        bif_df[f"Persona {idx+1}"] = list(bif_scores.values())

    # 점수 테이블 표출
    st.subheader("IPIP 점수 테이블")
    st.dataframe(ipip_df)

    st.subheader("BIF 점수 테이블")
    st.dataframe(bif_df)

    # 결과 확인 버튼
    if st.button("결과 확인"):
        st.write("선택된 점수:")
        for idx, persona in enumerate(personas):
            st.write(f"\nPersona {idx+1}:")
            st.write("IPIP 점수:", ipip_df[f"Persona {idx+1}"].to_dict())
            st.write("BIF 점수:", bif_df[f"Persona {idx+1}"].to_dict())

except FileNotFoundError as e:
    st.error(f"필요한 JSON 파일을 찾을 수 없습니다: {str(e)}")
except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")