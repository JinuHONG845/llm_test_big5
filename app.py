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

# 앱 기본 설정
st.set_page_config(
    page_title="Big 5 성격 검사",
    page_icon="🧪",
    layout="wide"
)

# 전역 변수로 test_mode 설정
test_mode = "전체 테스트 (분할 실행)"  # 기본값 설정

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

# 사이드바 설정
with st.sidebar:
    # 실험 모드 선택
    test_mode = st.radio(
        "실험 모드 선택",
        ("페르소나 테스트", "대조군 테스트"),
        horizontal=True
    )
    
    st.divider()  # 구분선 추가

    if test_mode == "페르소나 테스트":
        # === LLM 설정 섹션 ===
        st.title("LLM 설정")
        llm_choice = st.radio(
            "LLM 선택",
            ("GPT", "Claude", "Gemini"),
            horizontal=True,
            key="main_llm"
        )

        # LLM 선택에 따른 세부 모델 선택
        if llm_choice == "GPT":
            model_choice = st.radio(
                "GPT 모델 선택",
                ("GPT-4 Turbo", "GPT-3.5 Turbo"),
                horizontal=True,
                help="GPT-4 Turbo는 더 정확하지만 느립니다. GPT-3.5 Turbo는 더 빠르지만 정확도가 낮을 수 있습니다.",
                key="main_model_gpt"
            )
        elif llm_choice == "Claude":
            model_choice = st.radio(
                "Claude 모델 선택",
                ("Claude 3 Sonnet", "Claude 3 Haiku"),
                horizontal=True,
                help="Sonnet은 더 정확하지만 느립니다. Haiku는 더 빠르지만 정확도가 낮을 수 있습니다.",
                key="main_model_claude"
            )
        else:  # Gemini
            model_choice = st.radio(
                "Gemini 모델 선택",
                ("Gemini Pro",),  # 단일 옵션
                horizontal=True,
                help="현재 Gemini Pro 모델만 사용 가능합니다.",
                key="main_model_gemini"
            )
        
        # 대조군 변수 초기화
        control_llm_choice = None
        control_model_choice = None

    else:  # 대조군 테스트
        # === 대조군 LLM 설정 ===
        st.title("대조군 LLM 설정")
        
        # 대조군 LLM 선택
        control_llm_choice = st.radio(
            "LLM 선택",
            ("GPT", "Claude", "Gemini"),
            horizontal=True,
            key="control_llm"
        )

        # 대조군 세부 모델 선택
        if control_llm_choice == "GPT":
            control_model_choice = st.radio(
                "GPT 모델 선택",
                ("GPT-4 Turbo", "GPT-3.5 Turbo"),
                horizontal=True,
                help="GPT-4 Turbo는 더 정확하지만 느립니다. GPT-3.5 Turbo는 더 빠르지만 정확도가 낮을 수 있습니다.",
                key="control_model_gpt"
            )
        elif control_llm_choice == "Claude":
            control_model_choice = st.radio(
                "Claude 모델 선택",
                ("Claude 3 Sonnet", "Claude 3 Haiku"),
                horizontal=True,
                help="Sonnet은 더 정확하지만 느립니다. Haiku는 더 빠르지만 정확도가 낮을 수 있습니다.",
                key="control_model_claude"
            )
        else:  # Gemini
            control_model_choice = st.radio(
                "Gemini 모델 선택",
                ("Gemini Pro",),  # 단일 옵션
                horizontal=True,
                help="현재 Gemini Pro 모델만 사용 가능합니다.",
                key="control_model_gemini"
            )
        
        # LLM 변수 초기화
        llm_choice = None
        model_choice = None

# 세션 상태 초기화 (기존 코드에 추가)
if 'control_results' not in st.session_state:
    st.session_state.control_results = {
        'ipip': pd.DataFrame(),
        'bfi': pd.DataFrame(),
        'completed_batches': set()
    }

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

def run_batch_test(batch_name, start_idx, end_idx, test_type='IPIP', is_control=False):
    # 대조군 테스트일 경우 빈 페르소나 사용
    if is_control:
        personas_to_test = [{"personality": []}]  # 빈 페르소나
    else:
        personas_to_test = personas[start_idx:end_idx]  # 기존 페르소나
    
    ipip_batch_size, bfi_batch_size = get_batch_size(model_choice)
    
    # DataFrame 초기화 또는 기존 결과 불러오기
    if st.session_state.accumulated_results['ipip'].empty:
        ipip_df = pd.DataFrame(
            np.nan, 
            index=[f"Persona {i+1}" for i in range(len(personas))] + ['Average'],
            columns=[f"Q{i+1}" for i in range(300)]
        )
    else:
        ipip_df = st.session_state.accumulated_results['ipip'].copy()
    
    if st.session_state.accumulated_results['bfi'].empty:
        bfi_df = pd.DataFrame(
            np.nan, 
            index=[f"Persona {i+1}" for i in range(len(personas))] + ['Average'],
            columns=[f"Q{i+1}" for i in range(44)]
        )
    else:
        bfi_df = st.session_state.accumulated_results['bfi'].copy()

    # 진행 상황 표시를 위한 컨테이너 생성
    if test_type == 'IPIP':
        st.write("### IPIP 테스트 진행 상황")
        progress_bar = st.progress(0)
        result_table = st.empty()
        
        # IPIP 테스트 실행
        for i, persona in enumerate(personas_to_test, start=start_idx):
            # IPIP 테스트
            all_ipip_scores = []
            for j in range(0, 300, ipip_batch_size):
                try:
                    batch_end = min(j + ipip_batch_size, 300)
                    batch_questions = ipip_questions['items'][j:batch_end]
                    
                    ipip_responses = get_llm_response(persona, batch_questions, 'IPIP')
                    if ipip_responses and 'responses' in ipip_responses:
                        scores = [r['score'] for r in ipip_responses['responses']]
                        all_ipip_scores.extend(scores)
                        
                        current_scores = ipip_df.iloc[i].copy()
                        current_scores[j:j+len(scores)] = scores
                        ipip_df.iloc[i] = current_scores
                        ipip_df.loc['Average'] = ipip_df.iloc[:-1].mean()
                        
                        # 진행 상황 업데이트 (1.0을 초과하지 않도록 수정)
                        progress = min(1.0, ((i - start_idx) * 300 + j + len(scores)) / (len(personas_to_test) * 300))
                        progress_bar.progress(progress)
                        
                        # DataFrame 업데이트
                        result_table.dataframe(
                            ipip_df.fillna(0).round().astype(int).style
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
                        
                        time.sleep(1)  # 시각적 효과를 위한 짧은 대기
                        
                except Exception as e:
                    st.error(f"IPIP 테스트 오류 (페르소나 {i+1}, 문항 {j}-{batch_end}): {str(e)}")
                    continue
                
    else:  # BFI
        st.write("### BFI 테스트 진행 상황")
        progress_bar = st.progress(0)
        result_table = st.empty()
        
        # BFI 테스트 실행
        for i, persona in enumerate(personas_to_test, start=start_idx):
            # BFI 테스트 (유사한 방식으로 구현)
            for j in range(0, 44, bfi_batch_size):
                try:
                    batch_end = min(j + bfi_batch_size, 44)
                    batch_questions = bfi_questions[j:batch_end]
                    
                    bfi_responses = get_llm_response(persona, batch_questions, 'BFI')
                    if bfi_responses and 'responses' in bfi_responses:
                        scores = [r['score'] for r in bfi_responses['responses']]
                        
                        current_scores = bfi_df.iloc[i].copy()
                        current_scores[j:j+len(scores)] = scores
                        bfi_df.iloc[i] = current_scores
                        bfi_df.loc['Average'] = bfi_df.iloc[:-1].mean()
                        
                        # 진행 상황 업데이트 (1.0을 초과하지 않도록 수정)
                        progress = min(1.0, ((i - start_idx) * 44 + j + len(scores)) / (len(personas_to_test) * 44))
                        progress_bar.progress(progress)
                        
                        # DataFrame 업데이트
                        result_table.dataframe(
                            bfi_df.fillna(0).round().astype(int).style
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
                        
                        time.sleep(1)  # 시각적 효과를 위한 짧은 대기
                        
                except Exception as e:
                    st.error(f"BFI 테스트 오류 (페르소나 {i+1}, 문항 {j}-{batch_end}): {str(e)}")
                    continue

    # 결과 누적 저장
    st.session_state.accumulated_results['ipip'] = ipip_df
    st.session_state.accumulated_results['bfi'] = bfi_df
    st.session_state.accumulated_results['completed_batches'].add(batch_name)

    return ipip_df, bfi_df

# 배치 버튼 클릭 처리
if test_mode == "전체 테스트 (분할 실행)":
    if ipip_batch1:
        ipip_df, _ = run_batch_test('ipip_batch1', 0, 10, test_type='IPIP')
    elif ipip_batch2:
        ipip_df, _ = run_batch_test('ipip_batch2', 10, 20, test_type='IPIP')
    elif ipip_batch3:
        ipip_df, _ = run_batch_test('ipip_batch3', 20, 30, test_type='IPIP')
    elif ipip_batch4:
        ipip_df, _ = run_batch_test('ipip_batch4', 30, 40, test_type='IPIP')
    elif ipip_batch5:
        ipip_df, _ = run_batch_test('ipip_batch5', 40, 50, test_type='IPIP')
    elif bfi_batch1:
        _, bfi_df = run_batch_test('bfi_batch1', 0, 10, test_type='BFI')
    elif bfi_batch2:
        _, bfi_df = run_batch_test('bfi_batch2', 10, 20, test_type='BFI')
    elif bfi_batch3:
        _, bfi_df = run_batch_test('bfi_batch3', 20, 30, test_type='BFI')
    elif bfi_batch4:
        _, bfi_df = run_batch_test('bfi_batch4', 30, 40, test_type='BFI')
    elif bfi_batch5:
        _, bfi_df = run_batch_test('bfi_batch5', 40, 50, test_type='BFI')
elif test_mode == "간이 테스트 (랜덤 3개 페르소나)":
    # 랜덤 페르소나 선택
    random_personas = random.sample(personas, 3)
    # 간이 테스트 로직 실행
    # ... (기존 간이 테스트 코드) ...

# CSV 파일 생성 및 다운로드 부분
if not st.session_state.accumulated_results['ipip'].empty:
    csv_data = pd.concat([
        st.session_state.accumulated_results['ipip'].add_prefix('IPIP_Q'),
        st.session_state.accumulated_results['bfi'].add_prefix('BFI_Q')
    ], axis=1)
    
    st.download_button(
        label="결과 다운로드 (CSV)",
        data=csv_data.to_csv(index=True, float_format='%.10f'),
        file_name="personality_test_results.csv",
        mime="text/csv"
    )

# 대조군 테스트 실행 함수 수정
def run_control_batch_test(batch_name, start_idx, end_idx, test_type='IPIP'):
    empty_persona = {"personality": []}  # 빈 페르소나
    
    # DataFrame 초기화 또는 기존 결과 불러오기
    if st.session_state.control_results[test_type.lower()].empty:
        df = pd.DataFrame(
            np.nan,
            index=[f"Control {i+1}" for i in range(num_control_tests)] + ['Average'],
            columns=[f"Q{i+1}" for i in range(300 if test_type == 'IPIP' else 44)]
        )
    else:
        df = st.session_state.control_results[test_type.lower()].copy()

    # 진행 상황 표시
    st.write(f"### {test_type} 대조군 테스트 진행 상황")
    progress_bar = st.progress(0)
    result_table = st.empty()

    # 각 대조군 테스트 실행
    for test_num in range(num_control_tests):
        try:
            responses = get_llm_response(empty_persona, 
                                       ipip_questions['items'][start_idx:end_idx] if test_type == 'IPIP' else bfi_questions[start_idx:end_idx], 
                                       test_type)
            
            if responses and 'responses' in responses:
                scores = [r['score'] for r in responses['responses']]
                df.iloc[test_num, start_idx:end_idx] = scores
                df.loc['Average'] = df.iloc[:-1].mean()

                # 진행 상황 업데이트
                progress = (test_num + 1) / num_control_tests
                progress_bar.progress(progress)

                # 결과 테이블 업데이트
                result_table.dataframe(
                    df.fillna(0).round().astype(int).style
                        .background_gradient(cmap='YlOrRd', vmin=1, vmax=5)
                        .format("{:d}")
                        .set_properties(**{
                            'width': '40px',
                            'text-align': 'center',
                            'font-size': '13px',
                            'border': '1px solid #e6e6e6'
                        })
                )

        except Exception as e:
            st.error(f"대조군 테스트 {test_num+1} 실패: {str(e)}")
            continue

    # 결과 저장
    st.session_state.control_results[test_type.lower()] = df
    st.session_state.control_results['completed_batches'].add(batch_name)

    return df

# 메인 UI에서 대조군 테스트 버튼 처리
if test_mode == "대조군 테스트":
    st.write("### IPIP 대조군 테스트")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("IPIP 1-10", key="control_ipip_1"):
            run_batch_test('ipip_1', 0, 10, 'IPIP', is_control=True)
    with col2:
        if st.button("IPIP 11-20", key="control_ipip_2"):
            run_batch_test('ipip_2', 10, 20, 'IPIP', is_control=True)
    with col3:
        if st.button("IPIP 21-30", key="control_ipip_3"):
            run_batch_test('ipip_3', 20, 30, 'IPIP', is_control=True)
    with col4:
        if st.button("IPIP 31-40", key="control_ipip_4"):
            run_batch_test('ipip_4', 30, 40, 'IPIP', is_control=True)
    with col5:
        if st.button("IPIP 41-50", key="control_ipip_5"):
            run_batch_test('ipip_5', 40, 50, 'IPIP', is_control=True)

    st.write("### BFI 대조군 테스트")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("BFI 1-10", key="control_bfi_1"):
            run_batch_test('bfi_1', 0, 10, 'BFI', is_control=True)
    with col2:
        if st.button("BFI 11-20", key="control_bfi_2"):
            run_batch_test('bfi_2', 10, 20, 'BFI', is_control=True)
    with col3:
        if st.button("BFI 21-30", key="control_bfi_3"):
            run_batch_test('bfi_3', 20, 30, 'BFI', is_control=True)
    with col4:
        if st.button("BFI 31-40", key="control_bfi_4"):
            run_batch_test('bfi_4', 30, 40, 'BFI', is_control=True)
    with col5:
        if st.button("BFI 41-50", key="control_bfi_5"):
            run_batch_test('bfi_5', 40, 50, 'BFI', is_control=True)

    # 초기화 버튼
    if st.button("대조군 테스트 초기화", key="control_reset"):
        st.session_state.control_results = {
            'ipip': pd.DataFrame(),
            'bfi': pd.DataFrame(),
            'completed_batches': set()
        }
        st.rerun()

elif test_mode == "페르소나 테스트":
    st.write("### IPIP 페르소나 테스트")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("IPIP 1-10", key="persona_ipip_1"):
            run_batch_test('ipip_1', 0, 10, 'IPIP', is_control=False)
    with col2:
        if st.button("IPIP 11-20", key="persona_ipip_2"):
            run_batch_test('ipip_2', 10, 20, 'IPIP', is_control=False)
    with col3:
        if st.button("IPIP 21-30", key="persona_ipip_3"):
            run_batch_test('ipip_3', 20, 30, 'IPIP', is_control=False)
    with col4:
        if st.button("IPIP 31-40", key="persona_ipip_4"):
            run_batch_test('ipip_4', 30, 40, 'IPIP', is_control=False)
    with col5:
        if st.button("IPIP 41-50", key="persona_ipip_5"):
            run_batch_test('ipip_5', 40, 50, 'IPIP', is_control=False)

    st.write("### BFI 페르소나 테스트")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("BFI 1-10", key="persona_bfi_1"):
            run_batch_test('bfi_1', 0, 10, 'BFI', is_control=False)
    with col2:
        if st.button("BFI 11-20", key="persona_bfi_2"):
            run_batch_test('bfi_2', 10, 20, 'BFI', is_control=False)
    with col3:
        if st.button("BFI 21-30", key="persona_bfi_3"):
            run_batch_test('bfi_3', 20, 30, 'BFI', is_control=False)
    with col4:
        if st.button("BFI 31-40", key="persona_bfi_4"):
            run_batch_test('bfi_4', 30, 40, 'BFI', is_control=False)
    with col5:
        if st.button("BFI 41-50", key="persona_bfi_5"):
            run_batch_test('bfi_5', 40, 50, 'BFI', is_control=False)