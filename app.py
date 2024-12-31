import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
import anthropic
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 페이지 설정
st.set_page_config(layout="wide", page_title="LLM Big 5 Test")

# 공통 테이블 스타일 정의
TABLE_STYLE = {
    'properties': {
        'width': '40px',
        'text-align': 'center',
        'font-size': '13px',
        'border': '1px solid #e6e6e6'
    },
    'table_styles': [
        {'selector': 'th', 'props': [
            ('background-color', '#f0f2f6'),
            ('color', '#0e1117'),
            ('font-weight', 'bold'),
            ('text-align', 'center')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center')
        ]},
        {'selector': 'table', 'props': [
            ('width', '100%'),
            ('margin', '0 auto')
        ]}
    ]
}

def create_styled_table(df):
    """테이블 스타일링을 적용하는 헬퍼 함수"""
    return df.fillna(0).round().astype(int).style\
        .background_gradient(cmap='YlOrRd', vmin=1, vmax=5)\
        .format("{:d}")\
        .set_properties(**TABLE_STYLE['properties'])\
        .set_table_styles(TABLE_STYLE['table_styles'])

def initialize_table(title, columns, is_control=False):
    """테이블 초기화를 위한 헬퍼 함수"""
    st.write(f"### {title}")
    table = st.empty()
    prefix = "Control" if is_control else "Persona"
    df = pd.DataFrame(
        np.nan,
        index=[f"{prefix} {i+1}" for i in range(len(personas))] + [f'{prefix} Average'],
        columns=[f"Q{i+1}" for i in range(columns)]
    )
    table.dataframe(create_styled_table(df), use_container_width=True)
    return table

def initialize_tables():
    """모든 테이블 초기화"""
    if 'tables_initialized' not in st.session_state:
        st.session_state.ipip_table = initialize_table("IPIP 테스트 결과", 300)
        st.session_state.control_table = initialize_table("IPIP 대조군 테스트 결과", 300, True)
        st.session_state.bfi_table = initialize_table("BFI 테스트 결과", 44)
        st.session_state.bfi_control_table = initialize_table("BFI 대조군 테스트 결과", 44, True)
        st.session_state.tables_initialized = True

def create_test_buttons(prefix, disabled_check):
    """테스트 버튼 생성을 위한 헬퍼 함수"""
    col1, col2, col3, col4, col5 = st.columns(5)
    buttons = []
    for i, col in enumerate([col1, col2, col3, col4, col5], 1):
        with col:
            start_num = (i-1)*10 + 1
            end_num = i*10
            button = st.button(
                f"{prefix} {start_num}-{end_num}번",
                disabled=f'{prefix.lower()}_batch{i}' in disabled_check
            )
            buttons.append(button)
    return buttons

def get_batch_size(model):
    """모델에 따른 배치 크기 반환"""
    if model in ["GPT-4 Turbo", "Claude 3 Sonnet", "Gemini Pro"]:
        return 25, 5  # IPIP 배치 크기, BFI 배치 크기
    else:
        return 10, 3  # 더 작은 배치 크기

# JSON 파일 로드
try:
    personas = json.load(open('persona.json', 'r'))
    ipip_questions = json.load(open('IPIP.json', 'r'))['items']
    bfi_questions = json.load(open('BFI.json', 'r'))
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error(f"JSON 파일 로드 오류: {str(e)}")
    st.stop()

# 세션 상태 초기화
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = {
        'ipip': pd.DataFrame(),
        'bfi': pd.DataFrame(),
        'control_ipip': pd.DataFrame(),
        'bfi_control': pd.DataFrame(),
        'completed_batches': set()
    }

# LLM 선택
llm_choice = st.radio("LLM 선택", ("GPT", "Claude", "Gemini"), horizontal=True)
model_choice = st.radio(
    f"{llm_choice} 모델 선택",
    ("GPT-4 Turbo", "GPT-3.5 Turbo") if llm_choice == "GPT" else
    ("Claude 3 Sonnet", "Claude 3 Haiku") if llm_choice == "Claude" else
    ("Gemini Pro",),
    horizontal=True
)

# API 키 설정
api_keys = {
    "GPT": ("OPENAI_API_KEY", openai),
    "Claude": ("ANTHROPIC_API_KEY", lambda key: anthropic.Anthropic(api_key=key)),
    "Gemini": ("GOOGLE_API_KEY", lambda key: genai.configure(api_key=key))
}

api_key = st.secrets.get(api_keys[llm_choice][0])
if not api_key:
    st.error(f"{llm_choice} API 키가 설정되지 않았습니다.")
    st.stop()

if llm_choice == "GPT":
    openai.api_key = api_key
elif llm_choice == "Claude":
    client = api_keys[llm_choice][1](api_key)
else:
    api_keys[llm_choice][1](api_key)

# 테이블 초기화
initialize_tables()

# 테스트 버튼 생성
st.write("### IPIP 페르소나 배치 선택")
ipip_buttons = create_test_buttons("IPIP", st.session_state.accumulated_results['completed_batches'])

st.write("### IPIP 대조군 테스트 (페르소나 없음)")
control_buttons = create_test_buttons("Control", st.session_state.accumulated_results['completed_batches'])

st.write("### BFI 페르소나 배치 선택")
bfi_buttons = create_test_buttons("BFI", st.session_state.accumulated_results['completed_batches'])

st.write("### BFI 대조군 테스트 (페르소나 없음)")
bfi_control_buttons = create_test_buttons("BFI Control", st.session_state.accumulated_results['completed_batches'])

# 테스트 실행 함수들
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def get_llm_response(persona, questions, test_type):
    # LLM 응답 처리 로직
    pass  # 이 부분은 기존 코드를 유지

def run_batch_test(batch_name, start_idx, end_idx, test_type='IPIP'):
    # 배치 테스트 실행 로직
    pass  # 이 부분은 기존 코드를 유지

def run_control_batch_test(batch_name, start_idx, end_idx):
    # 대조군 테스트 실행 로직
    pass  # 이 부분은 기존 코드를 유지

# 버튼 클릭 처리
if any(ipip_buttons):
    idx = ipip_buttons.index(True)
    run_batch_test(f'ipip_batch{idx+1}', idx*10, (idx+1)*10, 'IPIP')

if any(control_buttons):
    idx = control_buttons.index(True)
    run_control_batch_test(f'control_batch{idx+1}', idx*10, (idx+1)*10)

if any(bfi_buttons):
    idx = bfi_buttons.index(True)
    run_batch_test(f'bfi_batch{idx+1}', idx*10, (idx+1)*10, 'BFI')

if any(bfi_control_buttons):
    idx = bfi_control_buttons.index(True)
    run_control_batch_test(f'bfi_control_batch{idx+1}', idx*10, (idx+1)*10)

# CSV 다운로드 버튼
if not st.session_state.accumulated_results['ipip'].empty:
    csv_data = pd.concat([
        st.session_state.accumulated_results['ipip'].add_prefix('IPIP_Q'),
        st.session_state.accumulated_results['bfi'].add_prefix('BFI_Q'),
        st.session_state.accumulated_results.get('control_ipip', pd.DataFrame()).add_prefix('Control_IPIP_Q'),
        st.session_state.accumulated_results.get('bfi_control', pd.DataFrame()).add_prefix('Control_BFI_Q')
    ], axis=1)
    
    st.download_button(
        label="결과 다운로드 (CSV)",
        data=csv_data.to_csv(index=True, float_format='%.10f'),
        file_name="personality_test_results.csv",
        mime="text/csv"
    )