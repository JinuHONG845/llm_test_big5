import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
import anthropic
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random  # 파일 상단에 추가

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

def update_table(table, df):
    """테이블 업데이트를 위한 헬퍼 함수"""
    table.dataframe(create_styled_table(df), use_container_width=True)

# 페이지 설정
st.set_page_config(layout="wide", page_title="LLM Big 5 Test")

# 여백 줄정을 위한 CSS
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 100rem;
        }
        .element-container {
            margin-bottom: 1.5rem;
        }
        .stDataFrame {
            width: 100%;
        }
        .dataframe {
            margin-top: 1rem;
            margin-bottom: 2rem;
            font-size: 14px;
        }
        table {
            border-collapse: collapse;
            border-spacing: 0;
            width: 100%;
        }
        th {
            background-color: #f0f2f6;
            font-weight: bold;
            padding: 12px 8px !important;
        }
        td {
            padding: 10px 8px !important;
        }
        h3 {
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
            padding: 0.5rem 0;
            border-bottom: 2px solid #f0f2f6;
            color: #0e1117;
        }
        .stProgress > div > div {
            background-color: #00cc99;
        }
    </style>
""", unsafe_allow_html=True)

# 기본 타이틀 설정
st.title("LLM Big 5 Test")

# JSON 파일들 로드
try:
    personas = json.load(open('persona.json', 'r'))
    ipip_questions = json.load(open('IPIP.json', 'r'))['items']
    bfi_questions = json.load(open('BFI.json', 'r'))
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error(f"JSON 파일 로드 오류: {str(e)}")
    st.stop()

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

# 세션 상태 초기화
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = {
        'ipip': pd.DataFrame(),
        'bfi': pd.DataFrame(),
        'control_ipip': pd.DataFrame(),
        'bfi_control': pd.DataFrame(),
        'completed_batches': set()
    }

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

# 배치 크기 조정 (모델에 따라)
def get_batch_size(model):
    if model in ["GPT-4 Turbo", "Claude 3 Sonnet", "Gemini Pro"]:
        return 25, 5  # IPIP 배치 크기, BFI 배치 크기
    else:
        return 10, 3  # 더 작은 배치 크기

def select_test_mode():
    print("\n전체 테스트를 시작합니다.")
    return "전체 테스트 (분할 실행)"

# test_mode 변수 정의
test_mode = select_test_mode()

def run_batch_test(batch_name, start_idx, end_idx, test_type='IPIP'):
    if test_type == 'IPIP':
        df_key = 'ipip'
        questions = ipip_questions
        total_questions = 300
        batch_size = get_batch_size(model_choice)[0]
        table_container = st.session_state.ipip_table
    else:  # BFI
        df_key = 'bfi'
        questions = bfi_questions
        total_questions = 44
        batch_size = get_batch_size(model_choice)[1]
        table_container = st.session_state.bfi_table

    # DataFrame 초기화 또는 기존 결과 불러오기
    if st.session_state.accumulated_results[df_key].empty:
        df = pd.DataFrame(
            np.nan, 
            index=[f"Persona {i+1}" for i in range(len(personas))] + ['Average'],
            columns=[f"Q{i+1}" for i in range(total_questions)]
        )
    else:
        df = st.session_state.accumulated_results[df_key].copy()

    # 진행 상황 표시
    progress_bar = st.progress(0)

    batch_personas = personas[start_idx:end_idx]
    for i, persona in enumerate(batch_personas, start=start_idx):
        for j in range(0, total_questions, batch_size):
            try:
                batch_end = min(j + batch_size, total_questions)
                batch_questions = questions[j:batch_end]
                
                responses = get_llm_response(persona, batch_questions, test_type)
                if responses and 'responses' in responses:
                    scores = [r['score'] for r in responses['responses']]
                    
                    current_scores = df.iloc[i].copy()
                    current_scores[j:j+len(scores)] = scores
                    df.iloc[i] = current_scores
                    df.loc['Average'] = df.iloc[:-1].mean()
                    
                    # 진행 상황 업데이트
                    progress = min(1.0, ((i - start_idx) * total_questions + j + len(scores)) / (len(batch_personas) * total_questions))
                    progress_bar.progress(progress)
                    
                    # DataFrame 업데이트
                    table_container.dataframe(
                        df.fillna(0).round().astype(int).style
                            .background_gradient(cmap='YlOrRd', vmin=1, vmax=5)
                            .format("{:d}")
                            .set_properties(**{
                                'width': '40px',
                                'text-align': 'center',
                                'font-size': '13px',
                                'border': '1px solid #e6e6e6'
                            })
                            .set_table_styles([
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
                            ]),
                        use_container_width=True
                    )
                    
                    time.sleep(1)
                    
            except Exception as e:
                st.error(f"{test_type} 테스트 오류 (페르소나 {i+1}, 문항 {j}-{batch_end}): {str(e)}")
                continue

    # 결과 저장
    st.session_state.accumulated_results[df_key] = df
    st.session_state.accumulated_results['completed_batches'].add(batch_name)

    return df

def run_control_batch_test(batch_name, start_idx, end_idx):
    """페르소나 없이 대조군 테스트를 실행하는 함수"""
    ipip_batch_size, _ = get_batch_size(model_choice)
    
    # DataFrame 초기화 또는 기존 결과 불러오기
    if 'control_ipip' not in st.session_state.accumulated_results:
        st.session_state.accumulated_results['control_ipip'] = pd.DataFrame(
            np.nan, 
            index=[f"Control {i+1}" for i in range(len(personas))] + ['Control Average'],
            columns=[f"Q{i+1}" for i in range(300)]
        )
    
    control_df = st.session_state.accumulated_results['control_ipip'].copy()
    
    # 진행 상황 표시
    st.write("### 대조군 IPIP 테스트 진행 상황")
    progress_bar = st.progress(0)
    result_table = st.empty()
    
    # 대조군 테스트 실행
    for i in range(start_idx, end_idx):
        all_control_scores = []
        for j in range(0, 300, ipip_batch_size):
            try:
                batch_end = min(j + ipip_batch_size, 300)
                batch_questions = ipip_questions[j:batch_end]
                
                # 페르소나 없이 테스트 실행
                empty_persona = {"personality": []}
                control_responses = get_llm_response(empty_persona, batch_questions, 'IPIP')
                
                if control_responses and 'responses' in control_responses:
                    scores = [r['score'] for r in control_responses['responses']]
                    all_control_scores.extend(scores)
                    
                    current_scores = control_df.iloc[i].copy()
                    current_scores[j:j+len(scores)] = scores
                    control_df.iloc[i] = current_scores
                    control_df.loc['Control Average'] = control_df.iloc[:-1].mean()
                    
                    # 진행 상황 업데이트
                    progress = min(1.0, ((i - start_idx) * 300 + j + len(scores)) / ((end_idx - start_idx) * 300))
                    progress_bar.progress(progress)
                    
                    # DataFrame 업데이트
                    result_table.dataframe(
                        control_df.fillna(0).round().astype(int).style
                            .background_gradient(cmap='YlOrRd', vmin=1, vmax=5)
                            .format("{:d}")
                            .set_properties(**{
                                'width': '40px',
                                'text-align': 'center',
                                'font-size': '13px',
                                'border': '1px solid #e6e6e6'
                            })
                            .set_table_styles([
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
                            ]),
                        use_container_width=True
                    )
                    
                    time.sleep(1)
                    
            except Exception as e:
                st.error(f"대조군 테스트 오류 (Control {i+1}, 문항 {j}-{batch_end}): {str(e)}")
                continue
    
    # 결과 저장
    st.session_state.accumulated_results['control_ipip'] = control_df
    st.session_state.accumulated_results['completed_batches'].add(batch_name)
    
    return control_df

# 배치 버튼 클릭 처리
if test_mode == "전체 테스트 (분할 실행)":
    # IPIP 테스트 배치
    if ipip_buttons[0]:
        run_batch_test('ipip_batch1', 0, 10, test_type='IPIP')
    if ipip_buttons[1]:
        run_batch_test('ipip_batch2', 10, 20, test_type='IPIP')
    if ipip_buttons[2]:
        run_batch_test('ipip_batch3', 20, 30, test_type='IPIP')
    if ipip_buttons[3]:
        run_batch_test('ipip_batch4', 30, 40, test_type='IPIP')
    if ipip_buttons[4]:
        run_batch_test('ipip_batch5', 40, 50, test_type='IPIP')

    # BFI 테스트 배치
    if bfi_buttons[0]:
        run_batch_test('bfi_batch1', 0, 10, test_type='BFI')
    if bfi_buttons[1]:
        run_batch_test('bfi_batch2', 10, 20, test_type='BFI')
    if bfi_buttons[2]:
        run_batch_test('bfi_batch3', 20, 30, test_type='BFI')
    if bfi_buttons[3]:
        run_batch_test('bfi_batch4', 30, 40, test_type='BFI')
    if bfi_buttons[4]:
        run_batch_test('bfi_batch5', 40, 50, test_type='BFI')

    # 대조군 테스트 배치
    if control_buttons[0]:
        run_control_batch_test('control_batch1', 0, 10)
    if control_buttons[1]:
        run_control_batch_test('control_batch2', 10, 20)
    if control_buttons[2]:
        run_control_batch_test('control_batch3', 20, 30)
    if control_buttons[3]:
        run_control_batch_test('control_batch4', 30, 40)
    if control_buttons[4]:
        run_control_batch_test('control_batch5', 40, 50)
elif test_mode == "간이 테스트 (랜덤 3개 페르소나)":
    # 랜덤 페르소나 선택
    random_personas = random.sample(personas, 3)
    # 간이 테스트 로직 실행
    # ... (기존 간이 테스트 코드) ...

# CSV 파일 생성 및 다운로드 부분
if not st.session_state.accumulated_results['ipip'].empty:
    csv_data = pd.concat([
        st.session_state.accumulated_results['ipip'].add_prefix('IPIP_Q'),
        st.session_state.accumulated_results['bfi'].add_prefix('BFI_Q'),
        st.session_state.accumulated_results.get('control_ipip', pd.DataFrame()).add_prefix('Control_IPIP_Q')
    ], axis=1)
    
    st.download_button(
        label="결과 다운로드 (CSV)",
        data=csv_data.to_csv(index=True, float_format='%.10f'),
        file_name="personality_test_results.csv",
        mime="text/csv"
    )