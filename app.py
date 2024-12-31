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

# LLM 선택 및 설정
col1, col2 = st.columns([1, 2])

with col1:
    llm_choice = st.radio(
        "LLM 선택",
        ("GPT", "Claude", "Gemini"),
        horizontal=True
    )

with col2:
    # LLM 선택에 따른 세부 모델 선택
    if llm_choice == "GPT":
        model_choice = st.radio(
            "GPT 모델 선택",
            ("GPT-4 Turbo", "GPT-3.5 Turbo"),
            horizontal=True,
            help="GPT-4 Turbo는 더 정확하지만 느립니다. GPT-3.5 Turbo는 더 빠르지만 정확도가 낮을 수 있습니다."
        )
    elif llm_choice == "Claude":
        model_choice = st.radio(
            "Claude 모델 선택",
            ("Claude 3 Sonnet", "Claude 3 Haiku"),
            horizontal=True,
            help="Sonnet은 더 정확하지만 느립니다. Haiku는 더 빠르지만 정확도가 낮을 수 있습니다."
        )
    else:  # Gemini
        model_choice = st.radio(
            "Gemini 모델 선택",
            ("Gemini Pro",),  # 단일 옵션
            horizontal=True,
            help="현재 Gemini Pro 모델만 사용 가능합니다."
        )

def select_test_mode():
    print("\n전체 테스트를 시작합니다.")
    return "전체 테스트 (분할 실행)"

# test_mode 변수 정의
test_mode = select_test_mode()

# API 키 설정
if llm_choice == "GPT":
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()
    openai.api_key = api_key
elif llm_choice == "Claude":
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Anthropic API 키가 설정되지 않았습니다.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)
else:  # Gemini
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API 키가 설정되지 않았습니다.")
        st.stop()
    genai.configure(api_key=api_key)

@retry(
    stop=stop_after_attempt(5),  # 최대 5번 재시도
    wait=wait_exponential(multiplier=1, min=4, max=20),  # 4~20초 사이 대기시간
    retry=(
        retry_if_exception_type(openai.APIError) |
        retry_if_exception_type(openai.APIConnectionError) |
        retry_if_exception_type(openai.RateLimitError) |
        retry_if_exception_type(anthropic.APIError) |
        retry_if_exception_type(anthropic.APIConnectionError) |
        retry_if_exception_type(anthropic.RateLimitError)
    )
)
def get_llm_response(persona, questions, test_type):
    """LLM을 사용하여 페르소나의 테스트 응답을 생성"""
    try:
        # 질문 목록 준비
        if test_type == 'IPIP':
            question_list = [q['item'] for q in questions]
            scale_description = """1 = Very inaccurate
2 = Moderately inaccurate
3 = Neither
4 = Moderately accurate
5 = Very accurate"""
        else:  # BFI
            question_list = [q['question'] for q in questions]
            scale_description = """1 = Disagree Strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly"""
        
        # 프롬프트 구성
        prompt = f"""Based on this persona: {', '.join(persona['personality'])}

For each question, provide a rating from 1-5 where:
{scale_description}

Return ONLY a JSON object in this exact format:
{{
    "responses": [
        {{"question": "<question text>", "score": <1-5>}},
        ...
    ]
}}

Questions to rate:
{json.dumps(question_list, indent=2)}"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if llm_choice == "GPT":
                    model_name = "gpt-4-turbo-preview" if model_choice == "GPT-4 Turbo" else "gpt-3.5-turbo"
                    response = openai.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.0,
                        timeout=30
                    )
                    content = response.choices[0].message.content
                    
                elif llm_choice == "Claude":
                    model_name = "claude-3-sonnet-20240229" if model_choice == "Claude 3 Sonnet" else "claude-3-haiku-20240307"
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=2000,
                        system="You are a helpful assistant that responds only in valid JSON format.",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        timeout=30
                    )
                    content = response.content[0].text
                    
                else:  # Gemini
                    model_name = 'gemini-pro' if model_choice == "Gemini Pro" else 'gemini-nano'
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=1.0
                        )
                    )
                    content = response.text
                
                # JSON 파싱 및 검증
                if content.startswith('```') and content.endswith('```'):
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                
                result = json.loads(content.strip())
                
                # 응답 검증
                if not result or 'responses' not in result:
                    raise ValueError("Invalid response format")
                if len(result['responses']) != len(question_list):
                    raise ValueError("Incomplete response")
                
                time.sleep(2)  # API 호출 사이에 2초 대기
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    st.error(f"응답 처리 중 오류 발생: {str(e)}")
                    raise e
                time.sleep(2 ** attempt)  # 지수 백오프
                continue
                
            except Exception as e:
                st.error(f"LLM API 오류: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
                continue
                
    except Exception as e:
        st.error(f"치명적인 오류 발생: {str(e)}")
        raise e

# 배치 크기 조정 (모델에 따라)
def get_batch_size(model):
    if model in ["GPT-4 Turbo", "Claude 3 Sonnet", "Gemini Pro"]:
        return 25, 5  # IPIP 배치 크기, BFI 배치 크기
    else:
        return 10, 3  # 더 작은 배치 크기

# 세션 상태 초기화
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = {
        'ipip': pd.DataFrame(),
        'bfi': pd.DataFrame(),
        'control_ipip': pd.DataFrame(),
        'completed_batches': set()
    }

if test_mode == "전체 테스트 (분할 실행)":
    # IPIP 테스트 섹션
    st.write("### IPIP 페르소나 배치 선택")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        ipip_batch1 = st.button("IPIP 1-10번", 
                          disabled='ipip_batch1' in st.session_state.accumulated_results['completed_batches'])
    with col2:
        ipip_batch2 = st.button("IPIP 11-20번", 
                          disabled='ipip_batch2' in st.session_state.accumulated_results['completed_batches'])
    with col3:
        ipip_batch3 = st.button("IPIP 21-30번", 
                          disabled='ipip_batch3' in st.session_state.accumulated_results['completed_batches'])
    with col4:
        ipip_batch4 = st.button("IPIP 31-40번", 
                          disabled='ipip_batch4' in st.session_state.accumulated_results['completed_batches'])
    with col5:
        ipip_batch5 = st.button("IPIP 41-50번", 
                          disabled='ipip_batch5' in st.session_state.accumulated_results['completed_batches'])

    # IPIP 결과 테이블
    st.write("### IPIP 테스트 결과")
    if 'ipip_table' not in st.session_state:
        st.session_state.ipip_table = st.empty()
        # 초기 빈 DataFrame 표시
        initial_ipip_df = pd.DataFrame(
            np.nan,
            index=[f"Persona {i+1}" for i in range(len(personas))] + ['Average'],
            columns=[f"Q{i+1}" for i in range(300)]
        )
        st.session_state.ipip_table.dataframe(
            initial_ipip_df.style
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

    # 대조군 테스트 섹션 추가
    st.write("### IPIP 대조군 테스트 (페르소나 없음)")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        control_batch1 = st.button("대조군 1-10번", 
                          disabled='control_batch1' in st.session_state.accumulated_results['completed_batches'])
    with col2:
        control_batch2 = st.button("대조군 11-20번", 
                          disabled='control_batch2' in st.session_state.accumulated_results['completed_batches'])
    with col3:
        control_batch3 = st.button("대조군 21-30번", 
                          disabled='control_batch3' in st.session_state.accumulated_results['completed_batches'])
    with col4:
        control_batch4 = st.button("대조군 31-40번", 
                          disabled='control_batch4' in st.session_state.accumulated_results['completed_batches'])
    with col5:
        control_batch5 = st.button("대조군 41-50번", 
                          disabled='control_batch5' in st.session_state.accumulated_results['completed_batches'])

    # 대조군 결과 테이블
    st.write("### 대조군 테스트 결과")
    if 'control_table' not in st.session_state:
        st.session_state.control_table = st.empty()
        # 초기 빈 DataFrame 표시
        initial_control_df = pd.DataFrame(
            np.nan,
            index=[f"Control {i+1}" for i in range(len(personas))] + ['Control Average'],
            columns=[f"Q{i+1}" for i in range(300)]
        )
        st.session_state.control_table.dataframe(
            initial_control_df.style
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

    # BFI 테스트 섹션
    st.write("### BFI 페르소나 배치 선택")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        bfi_batch1 = st.button("BFI 1-10번", 
                          disabled='bfi_batch1' in st.session_state.accumulated_results['completed_batches'])
    with col2:
        bfi_batch2 = st.button("BFI 11-20번", 
                          disabled='bfi_batch2' in st.session_state.accumulated_results['completed_batches'])
    with col3:
        bfi_batch3 = st.button("BFI 21-30번", 
                          disabled='bfi_batch3' in st.session_state.accumulated_results['completed_batches'])
    with col4:
        bfi_batch4 = st.button("BFI 31-40번", 
                          disabled='bfi_batch4' in st.session_state.accumulated_results['completed_batches'])
    with col5:
        bfi_batch5 = st.button("BFI 41-50번", 
                          disabled='bfi_batch5' in st.session_state.accumulated_results['completed_batches'])

    # 초기화 버튼
    if st.button("테스트 초기화"):
        st.session_state.accumulated_results = {
            'ipip': pd.DataFrame(),
            'bfi': pd.DataFrame(),
            'completed_batches': set()
        }
        st.rerun()

def run_batch_test(batch_name, start_idx, end_idx, test_type='IPIP'):
    if test_type == 'IPIP':
        df_key = 'ipip'
        questions = ipip_questions['items']
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
                batch_questions = ipip_questions['items'][j:batch_end]
                
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
    if ipip_batch1:
        run_batch_test('ipip_batch1', 0, 10, test_type='IPIP')
    if ipip_batch2:
        run_batch_test('ipip_batch2', 10, 20, test_type='IPIP')
    if ipip_batch3:
        run_batch_test('ipip_batch3', 20, 30, test_type='IPIP')
    if ipip_batch4:
        run_batch_test('ipip_batch4', 30, 40, test_type='IPIP')
    if ipip_batch5:
        run_batch_test('ipip_batch5', 40, 50, test_type='IPIP')

    # BFI 테스트 배치
    if bfi_batch1:
        run_batch_test('bfi_batch1', 0, 10, test_type='BFI')
    if bfi_batch2:
        run_batch_test('bfi_batch2', 10, 20, test_type='BFI')
    if bfi_batch3:
        run_batch_test('bfi_batch3', 20, 30, test_type='BFI')
    if bfi_batch4:
        run_batch_test('bfi_batch4', 30, 40, test_type='BFI')
    if bfi_batch5:
        run_batch_test('bfi_batch5', 40, 50, test_type='BFI')

    # 대조군 테스트 배치
    if control_batch1:
        run_control_batch_test('control_batch1', 0, 10)
    if control_batch2:
        run_control_batch_test('control_batch2', 10, 20)
    if control_batch3:
        run_control_batch_test('control_batch3', 20, 30)
    if control_batch4:
        run_control_batch_test('control_batch4', 30, 40)
    if control_batch5:
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