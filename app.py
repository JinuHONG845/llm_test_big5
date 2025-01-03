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
import re

# 앱 기본 설정
st.set_page_config(
    page_title="Big 5 성격 검사",
    page_icon="🧪",
    layout="wide"
)

# 전역 변수로 test_mode 설정 제거
# test_mode = "전체 테스트 (분할 실행)"  # 이 줄 삭제

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

# LLM 선택 및 설정 부분을 사이드바로 이동
with st.sidebar:
    st.title("테스트 설정")
    
    # 테스트 모드 선택 추가
    test_mode = st.radio(
        "테스트 모드 선택",
        ("페르소나 테스트", "대조군 테스트"),
        help="페르소나 테스트: 정의된 페르소나로 테스트 수행\n대조군 테스트: LLM이 자율적으로 응답"
    )
    
    st.title("LLM 설정")
    llm_choice = st.radio(
        "LLM 선택",
        ("GPT", "Claude", "Gemini"),
        horizontal=True
    )

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
    """LLM을 사용하여 페스트 응답을 생성"""
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
        
        # 프롬프트 구성 - 테스트 모드에 따라 다르게
        if test_mode == "페르소나 테스트":
            prompt = f"""Based on this persona: {', '.join(persona['personality'])}

For each question, provide a rating from 1-5 where:
{scale_description}

Return ONLY comma-separated numbers (e.g., 4,2,5,1,3). No other text or explanation.

Questions to rate:
{json.dumps(question_list, indent=2)}"""
        else:  # 대조군 테스트
            prompt = f"""As an AI, please answer these personality test questions honestly.
Rate each question from 1-5 where:
{scale_description}

Return ONLY comma-separated numbers (e.g., 4,2,5,1,3). No other text or explanation.

Questions to rate:
{json.dumps(question_list, indent=2)}"""

        # 시스템 메시지를 더 명확하게 수정
        system_message = "You must respond with ONLY comma-separated numbers (e.g., 4,2,5,1,3). Do not include any other text, JSON, or formatting."
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if llm_choice == "GPT":
                    model_name = "gpt-4-turbo-preview" if model_choice == "GPT-4 Turbo" else "gpt-3.5-turbo"
                    response = openai.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_message},
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
                        system=system_message,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        timeout=30
                    )
                    content = response.content[0].text
                    
                else:  # Gemini
                    model_name = 'gemini-pro' if model_choice == "Gemini Pro" else 'gemini-nano'
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        f"{system_message}\n\n{prompt}",
                        generation_config=genai.types.GenerationConfig(
                            temperature=1.0
                        )
                    )
                    content = response.text

                # 응답 정제 강화
                content = content.strip()
                
                # 모든 불필요한 문자 제거
                # JSON, 대괄호, 중괄호 등 제거
                content = re.sub(r'[\[\]{}"\'\n\r]', '', content)
                
                # "rating:", "response:", 등의 텍스트 제거
                content = re.sub(r'[a-zA-Z:]+', '', content)
                
                # 연속된 쉼표를 하나로
                content = re.sub(r',+', ',', content)
                
                # 앞뒤 쉼표 제거
                content = content.strip(',')
                
                # 숫자와 쉼표만 남기고 모두 제거
                content = re.sub(r'[^0-9,]', '', content)
                
                # 빈 문자열 체크
                if not content:
                    raise ValueError("No valid numbers found in response")
                
                # 숫자 추출 및 검증
                scores = [int(x.strip()) for x in content.split(',')]
                
                # 점수 범위 검증 (1-5)
                if not all(1 <= score <= 5 for score in scores):
                    raise ValueError("Scores must be between 1 and 5")
                
                # 응답 수 검증
                if len(scores) != len(question_list):
                    raise ValueError(f"Expected {len(question_list)} responses, got {len(scores)}")
                
                # 기존 형식으로 변환
                result = {
                    "responses": [
                        {"question": q, "score": s} for q, s in zip(question_list, scores)
                    ]
                }
                
                time.sleep(2)
                return result
                
            except Exception as e:
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
        return 25, 10  # IPIP 배치 크기, BFI 배치 크기 (5에서 10으로 증가)
    else:
        return 15, 8  # 더 작은 모델용 배치 크기도 증가

# 세션 상태 초기화
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = {
        'ipip': pd.DataFrame(),
        'bfi': pd.DataFrame(),
        'completed_batches': set()
    }

# IPIP 테스트 섹션
st.write("### IPIP 배치 선택")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ipip_batch1 = st.button("IPIP 1-10", 
                      disabled='ipip_batch1' in st.session_state.accumulated_results['completed_batches'])
with col2:
    ipip_batch2 = st.button("IPIP 11-20", 
                      disabled='ipip_batch2' in st.session_state.accumulated_results['completed_batches'])
with col3:
    ipip_batch3 = st.button("IPIP 21-30", 
                      disabled='ipip_batch3' in st.session_state.accumulated_results['completed_batches'])
with col4:
    ipip_batch4 = st.button("IPIP 31-40", 
                      disabled='ipip_batch4' in st.session_state.accumulated_results['completed_batches'])
with col5:
    ipip_batch5 = st.button("IPIP 41-50", 
                      disabled='ipip_batch5' in st.session_state.accumulated_results['completed_batches'])

# BFI 테스트 섹션
st.write("### BFI 배치 선택")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    bfi_batch1 = st.button("BFI 1-10", 
                      disabled='bfi_batch1' in st.session_state.accumulated_results['completed_batches'])
with col2:
    bfi_batch2 = st.button("BFI 11-20", 
                      disabled='bfi_batch2' in st.session_state.accumulated_results['completed_batches'])
with col3:
    bfi_batch3 = st.button("BFI 21-30", 
                      disabled='bfi_batch3' in st.session_state.accumulated_results['completed_batches'])
with col4:
    bfi_batch4 = st.button("BFI 31-40", 
                      disabled='bfi_batch4' in st.session_state.accumulated_results['completed_batches'])
with col5:
    bfi_batch5 = st.button("BFI 41-50", 
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
    ipip_batch_size, bfi_batch_size = get_batch_size(model_choice)
    
    if test_mode == "페르소나 테스트":
        batch_personas = personas[start_idx:end_idx]
        index_prefix = "Persona"
    else:  # 대조군 테스트
        batch_personas = [{"personality": ["AI Baseline Test"]} for _ in range(end_idx - start_idx)]
        index_prefix = "Dummy"

    # DataFrame 초기화 또는 기존 결과 불러오기
    if st.session_state.accumulated_results['ipip'].empty:
        ipip_df = pd.DataFrame(
            np.nan, 
            index=[f"{index_prefix} {i+1}" for i in range(len(personas))] + ['Average'],
            columns=[f"Q{i+1}" for i in range(300)]
        )
    else:
        ipip_df = st.session_state.accumulated_results['ipip'].copy()
    
    if st.session_state.accumulated_results['bfi'].empty:
        bfi_df = pd.DataFrame(
            np.nan, 
            index=[f"{index_prefix} {i+1}" for i in range(len(personas))] + ['Average'],
            columns=[f"Q{i+1}" for i in range(44)]  # BFI는 44문항
        )
    else:
        bfi_df = st.session_state.accumulated_results['bfi'].copy()

    # 진행 상황 표시를 위한 컨테이너 생성
    if test_type == 'IPIP':
        st.write("### IPIP 테스트 진행 상황")
        progress_bar = st.progress(0)
        result_table = st.empty()
        
        # IPIP 테스트 실행
        for i, persona in enumerate(batch_personas, start=start_idx):
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
                        progress = min(1.0, ((i - start_idx) * 300 + j + len(scores)) / (len(batch_personas) * 300))
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
        
        total_questions = 44
        max_retries = 3
        
        for i, persona in enumerate(batch_personas, start=start_idx):
            all_bfi_scores = []
            current_question = 0
            
            while current_question < total_questions:
                batch_end = min(current_question + bfi_batch_size, total_questions)
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        current_batch = bfi_questions[current_question:batch_end]
                        bfi_responses = get_llm_response(persona, current_batch, 'BFI')
                        
                        if bfi_responses and 'responses' in bfi_responses:
                            scores = [r['score'] for r in bfi_responses['responses']]
                            
                            if len(scores) == len(current_batch):
                                for idx, score in enumerate(scores):
                                    col_name = f"Q{current_question+idx+1}"
                                    bfi_df.at[f"{index_prefix} {i+1}", col_name] = score
                                
                                all_bfi_scores.extend(scores)
                                bfi_df.loc['Average'] = bfi_df.iloc[:-1].mean()
                                
                                progress = min(1.0, ((i - start_idx) * total_questions + len(all_bfi_scores)) / 
                                            ((end_idx - start_idx) * total_questions))
                                progress_bar.progress(progress)
                                
                                # DataFrame 업데이트 빈도 줄이기 (매 5문항마다)
                                if len(all_bfi_scores) % 5 == 0:
                                    result_table.dataframe(
                                        bfi_df.fillna(0).round().astype(int).style
                                            .background_gradient(cmap='YlOrRd', vmin=1, vmax=5)
                                            .format("{:d}"),
                                        use_container_width=True
                                    )
                                
                                time.sleep(1)  # 대기 시간 2초에서 1초로 감소
                                break
                                
                            else:
                                raise ValueError("응답 수 불일치")
                                
                        else:
                            raise ValueError("유효하지 않은 응답 형식")
                            
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"문항 처리 실패: {current_question+1}-{batch_end}")
                        time.sleep(retry_count)  # 지수 백오프 대신 선형 백오프 사용
                        continue
                
                current_question = batch_end

    # 결과 누적 저장
    st.session_state.accumulated_results['ipip'] = ipip_df
    st.session_state.accumulated_results['bfi'] = bfi_df
    st.session_state.accumulated_results['completed_batches'].add(batch_name)

    return ipip_df, bfi_df

# 배치 버튼 클릭 처리
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
    _, bfi_df = run_batch_test('bfi_batch1', 0, 10, test_type='BFI')  # 1-10번 더미/페르소나
elif bfi_batch2:
    _, bfi_df = run_batch_test('bfi_batch2', 10, 20, test_type='BFI')  # 11-20번 더미/페르소나
elif bfi_batch3:
    _, bfi_df = run_batch_test('bfi_batch3', 20, 30, test_type='BFI')  # 21-30번 더미/페르소나
elif bfi_batch4:
    _, bfi_df = run_batch_test('bfi_batch4', 30, 40, test_type='BFI')  # 31-40번 더미/페르소나
elif bfi_batch5:
    _, bfi_df = run_batch_test('bfi_batch5', 40, 50, test_type='BFI')  # 41-50번 더미/페르소나

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