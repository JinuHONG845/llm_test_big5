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
col1, col2 = st.columns([2, 3])
with col1:
    llm_choice = st.radio("LLM 선택", ("GPT-4", "Claude 3", "Gemini Pro"), horizontal=True)

with col2:
    test_mode = st.radio(
        "테스트 모드 선택",
        ("전체 테스트", "간이 테스트 (랜덤 3개 페르소나)"),
        horizontal=True,
        help="간이 테스트는 전체 페르소나 중 무작위로 3개를 선택하여 진행합니다."
    )

# API 키 설정
if llm_choice == "GPT-4":
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()
    openai.api_key = api_key
elif llm_choice == "Claude 3":
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Anthropic API 키가 설정되지 않았습니다.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)
else:  # Gemini Pro
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
                if llm_choice == "GPT-4":
                    response = openai.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.0,
                        timeout=30  # 30초 타임아웃 설정
                    )
                    content = response.choices[0].message.content
                    
                elif llm_choice == "Claude 3":
                    response = client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=2000,
                        system="You are a helpful assistant that responds only in valid JSON format.",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        timeout=30  # 30초 타임아웃 설정
                    )
                    content = response.content[0].text
                    
                else:  # Gemini Pro
                    model = genai.GenerativeModel('gemini-pro')
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

# 테스트 실행 버튼
if st.button("테스트 시작"):
    all_results = {}
    
    # 테스트할 페르소나 선택
    if test_mode == "간이 테스트 (랜덤 3개 페르소나)":
        test_personas = random.sample(personas, 3)
    else:
        test_personas = personas
    
    # 선택된 페르소나 정보 표시
    if test_mode == "간이 테스트 (랜덤 3개 페르소나)":
        st.write("### 선택된 페르소나")
        for i, persona in enumerate(test_personas, 1):
            st.write(f"페르소나 {i}: {', '.join(persona['personality'])}")
        st.write("---")
    
    # 빈이터프레임 초기화
    ipip_df = pd.DataFrame(
        np.nan, 
        index=[f"Persona {i+1}" for i in range(len(test_personas))] + ['Average'],
        columns=[f"Q{i+1}" for i in range(300)]
    )
    bfi_df = pd.DataFrame(
        np.nan, 
        index=[f"Persona {i+1}" for i in range(len(test_personas))] + ['Average'],
        columns=[f"Q{i+1}" for i in range(44)]
    )
    
    # CSV 저장용 데이터프레임
    ipip_df_full = ipip_df.copy()
    bfi_df_full = bfi_df.copy()
    
    # IPIP 테스트 섹션
    st.markdown("---")  # 구분선 추가
    st.markdown("""
        <h3 style='text-align: center; color: #0e1117; background-color: #f0f2f6; 
        padding: 1rem; border-radius: 5px;'>IPIP Test</h3>
    """, unsafe_allow_html=True)
    ipip_progress = st.progress(0)
    ipip_table = st.empty()
    
    # 초기 IPIP 테이블 표시
    ipip_table.dataframe(
        ipip_df.fillna(0).round().astype(int).style
            .background_gradient(
                cmap='YlOrRd',
                vmin=1,
                vmax=5
            )
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
    
    # BFI 테스트 섹션
    st.markdown("---")  # 구분선 추가
    st.markdown("""
        <h3 style='text-align: center; color: #0e1117; background-color: #f0f2f6; 
        padding: 1rem; border-radius: 5px;'>BFI Test</h3>
    """, unsafe_allow_html=True)
    bfi_progress = st.progress(0)
    bfi_table = st.empty()
    
    # 초기 BFI 테이블 표시
    bfi_table.dataframe(
        bfi_df.fillna(0).round().astype(int).style
            .background_gradient(
                cmap='YlOrRd',
                vmin=1,
                vmax=5
            )
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
    
    # IPIP 테스트 실행
    for i, persona in enumerate(test_personas):
        all_ipip_scores = []
        # 배치 크기를 25개로 줄임
        for j in range(0, 300, 25):  
            try:
                batch_end = min(j + 25, 300)  # 마지막 배치 처리
                batch_questions = ipip_questions['items'][j:batch_end]
                
                # 재시도 횟수 증가
                for retry_count in range(3):  
                    try:
                        ipip_responses = get_llm_response(persona, batch_questions, 'IPIP')
                        if ipip_responses and 'responses' in ipip_responses:
                            scores = [r['score'] for r in ipip_responses['responses']]
                            all_ipip_scores.extend(scores)
                            
                            current_scores = ipip_df.iloc[i].copy()
                            current_scores[j:j+len(scores)] = scores
                            ipip_df.iloc[i] = current_scores
                            ipip_df.loc['Average'] = ipip_df.iloc[:-1].mean()
                            
                            ipip_df_full.iloc[i] = current_scores
                            ipip_df_full.loc['Average'] = ipip_df_full.iloc[:-1].mean()
                            
                            progress = (i * 300 + j + len(scores)) / (len(test_personas) * 300)
                            ipip_progress.progress(progress)
                            
                            ipip_table.dataframe(
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
                            
                            time.sleep(3)  # API 호출 간격 증가
                            break  # 성공시 재시도 루프 종료
                            
                    except Exception as e:
                        if retry_count == 2:  # 마지막 시도였을 경우
                            st.error(f"IPIP 배치 처리 중 오류 발생 (페르소나 {i+1}, 문항 {j}-{batch_end}): {str(e)}")
                            raise e
                        time.sleep(5 * (retry_count + 1))  # 재시도 간격 증가
                        continue
                        
            except Exception as e:
                st.error(f"IPIP 테스트 중단 (페르소나 {i+1}): {str(e)}")
                raise e

    st.write("IPIP Test 완료")
    
    # BFI 테스트 실행 - 배치 크기를 더 작게 조정
    for i, persona in enumerate(test_personas):
        for j in range(0, 44, 5):  # 5개씩 배치 처리
            try:
                batch_end = min(j + 5, 44)
                batch_questions = bfi_questions[j:batch_end]
                
                # 재시도 횟수 증가
                for retry_count in range(3):
                    try:
                        bfi_responses = get_llm_response(persona, batch_questions, 'BFI')
                        if bfi_responses and 'responses' in bfi_responses:
                            scores = [r['score'] for r in bfi_responses['responses']]
                            current_scores = bfi_df.iloc[i].copy()
                            current_scores[j:j+len(scores)] = scores
                            bfi_df.iloc[i] = current_scores
                            bfi_df.loc['Average'] = bfi_df.iloc[:-1].mean()
                            
                            bfi_df_full.iloc[i] = current_scores
                            bfi_df_full.loc['Average'] = bfi_df_full.iloc[:-1].mean()
                            
                            progress = (i * 44 + batch_end) / (len(test_personas) * 44)
                            bfi_progress.progress(progress)
                            
                            bfi_table.dataframe(
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
                            
                            time.sleep(3)  # API 호출 간격 증가
                            break  # 성공시 재시도 루프 종료
                            
                    except Exception as e:
                        if retry_count == 2:  # 마지막 시도였을 경우
                            st.error(f"BFI 배치 처리 중 오류 발생 (페르소나 {i+1}, 문항 {j}-{batch_end}): {str(e)}")
                            raise e
                        time.sleep(5 * (retry_count + 1))  # 재시도 간격 증가
                        continue
                        
            except Exception as e:
                st.error(f"BFI 테스트 중단 (페르소나 {i+1}): {str(e)}")
                raise e
    
    st.write("BFI Test 완료")
    
    # CSV 파일 생성
    csv_data = pd.concat([
        ipip_df_full.add_prefix('IPIP_Q'),
        bfi_df_full.add_prefix('BFI_Q')
    ], axis=1)
    
    # 결과 다운로드 버튼
    st.download_button(
        label="결과 다운로드 (CSV)",
        data=csv_data.to_csv(index=True, float_format='%.10f'),
        file_name="personality_test_results.csv",
        mime="text/csv"
    )
    
    st.success("모든 테스트가 완료되었습니다!")