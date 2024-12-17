import streamlit as st
import pandas as pd
import json

# 기본 타이틀 설정
st.title("성격 평가 테스트")

# JSON 파일들 로드
try:
    with open('persona.json', 'r') as f:
        personas = json.load(f)
    with open('IPIP.json', 'r') as f:
        ipip_questions = json.load(f)
    with open('BFI.json', 'r') as f:
        bfi_questions = json.load(f)
except FileNotFoundError as e:
    st.error(f"필요한 JSON 파일을 찾을 수 없습니다: {str(e)}")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"JSON 파일 파싱 오류: {str(e)}")
    st.stop()

# LLM 선택 버튼
llm_choice = st.radio("LLM 선택", ("GPT-40", "Claude Sonnet 3.5", "Gemini Pro"))

# 세션 스테이트 초기화
if 'current_persona' not in st.session_state:
    st.session_state.current_persona = 0
if 'test_phase' not in st.session_state:
    st.session_state.test_phase = 'IPIP'  # 'IPIP' 또는 'BFI'
if 'all_scores' not in st.session_state:
    st.session_state.all_scores = {}

# 현재 페르소나 표시
st.write(f"### 현재 페르소나 {st.session_state.current_persona + 1}/50")
st.write("### 페르소나 특성:")
for trait in personas[st.session_state.current_persona]["personality"]:
    st.write(f"- {trait}")

# 현재 테스트 단계 표시
st.write(f"### 현재 테스트: {st.session_state.test_phase}")

# 현재 테스트에 따른 질문 설정
questions = ipip_questions if st.session_state.test_phase == 'IPIP' else bfi_questions

# 점수 입력 폼
scores = {}
with st.form(key=f'test_form_{st.session_state.current_persona}_{st.session_state.test_phase}'):
    for question in questions:
        question_text = question.get('item', question.get('question', ''))
        score = st.slider(
            question_text,
            min_value=1,
            max_value=5,
            value=3,
            help="1=매우 그렇지 않다, 2=그렇지 않다, 3=보통이다, 4=그렇다, 5=매우 그렇다"
        )
        scores[question_text] = score
    
    submit_button = st.form_submit_button("다음")

# 폼 제출 처리
if submit_button:
    # 현재 점수 저장
    persona_key = f"persona_{st.session_state.current_persona + 1}"
    if persona_key not in st.session_state.all_scores:
        st.session_state.all_scores[persona_key] = {}
    st.session_state.all_scores[persona_key][st.session_state.test_phase] = scores
    
    # 다음 단계로 이동
    if st.session_state.test_phase == 'IPIP':
        st.session_state.test_phase = 'BFI'
    else:
        st.session_state.test_phase = 'IPIP'
        st.session_state.current_persona += 1
    
    # 모든 테스트가 완료되었는지 확인
    if st.session_state.current_persona >= len(personas):
        st.write("### 모든 테스트가 완료되었습니다!")
        
        # 결과 다운로드 버튼
        results_df = pd.DataFrame.from_dict(st.session_state.all_scores, orient='index')
        st.download_button(
            label="결과 다운로드 (CSV)",
            data=results_df.to_csv(index=True),
            file_name="personality_test_results.csv",
            mime="text/csv"
        )
        
        # 세션 초기화 버튼
        if st.button("테스트 다시 시작"):
            st.session_state.current_persona = 0
            st.session_state.test_phase = 'IPIP'
            st.session_state.all_scores = {}
            st.experimental_rerun()
    else:
        st.experimental_rerun()

# 진행 상황 표시
progress = (st.session_state.current_persona * 2 + (1 if st.session_state.test_phase == 'BFI' else 0)) / (len(personas) * 2)
st.progress(progress)
st.write(f"진행률: {progress*100:.1f}%")