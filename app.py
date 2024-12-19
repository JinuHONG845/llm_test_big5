import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
import anthropic
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import random  # 파일 상단에 추가

# 페이지 설정
st.set_page_config(layout="wide", page_title="LLM Big 5 Test")

# 여백 줄정을 위한 CSS
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 3rem;  /* 좌측 여백 증가 */
            padding-right: 3rem;  /* 우측 여백 증가 */
            max-width: 100rem;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        .stDataFrame {
            width: 100%;
        }
        /* 테이블 헤더와 내용 사이 간격 */
        .dataframe {
            margin-top: 0.5rem;
            margin-bottom: 2rem;
        }
        /* 제목과 테이블 사이 간격 */
        h3 {
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
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
    stop=stop_after_attempt(3),  # 최대 3번 재시도
    wait=wait_exponential(multiplier=1, min=4, max=10),  # 4~10초 사이 대기시간
    retry_error_callback=lambda retry_state: None  # 실패시 None 반환
)
def get_llm_response(persona, questions, test_type):
    """LLM을 사용하여 페르소나의 테스트 응답을 생성"""
    
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
{json.dumps(question_list[:50], indent=2)}"""
    
    try:
        if llm_choice == "GPT-4":
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0
            )
            content = response.choices[0].message.content
            if content.startswith('```') and content.endswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            result = json.loads(content.strip())
            
            # 응답 검증
            if not result or 'responses' not in result or len(result['responses']) < len(question_list):
                raise ValueError("불완전한 응답")
                
            time.sleep(2)  # API 호출 사이에 2초 대기
            return result
        
        elif llm_choice == "Claude 3":
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system="You are a helpful assistant that responds only in valid JSON format.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1.0
            )
            result = json.loads(response.content[0].text)
            
            # 응답 검증
            if not result or 'responses' not in result or len(result['responses']) < len(question_list):
                raise ValueError("불완전한 응답")
                
            time.sleep(2)  # API 호출 사이에 2초 대기
            return result
        
        else:  # Gemini Pro
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=1.0
                )
            )
            try:
                return json.loads(response.text)
            except:
                # JSON 형식이 아닌 경우 응답에서 JSON 부분만 추출 
                import re
                json_str = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_str:
                    return json.loads(json_str.group())
                raise Exception("Invalid JSON response")
            
    except Exception as e:
        st.error(f"LLM API 오류: {str(e)}")
        st.write("프롬프트:", prompt)
        st.write("응답:", response if 'response' in locals() else "No response")
        raise e  # 재시도를 위해 예외를 다시 발생

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
    st.write("### IPIP Test")
    ipip_progress = st.progress(0)
    ipip_table = st.empty()
    
    # 초기 IPIP 테이블 표시
    ipip_table.dataframe(
        ipip_df.fillna(0).round().astype(int).style
            .background_gradient(cmap='YlOrRd')
            .format("{:d}")
            .set_properties(**{'width': '40px'})
            .set_table_styles([
                {'selector': 'table', 'props': [('width', '100%')]},
            ]),
        use_container_width=True
    )
    
    # BFI 테스트 섹션 (세로로 배치)
    st.write("### BFI Test")
    bfi_progress = st.progress(0)
    bfi_table = st.empty()
    
    # 초기 BFI 테이블 표시
    bfi_table.dataframe(
        bfi_df.fillna(0).round().astype(int).style
            .background_gradient(cmap='YlOrRd')
            .format("{:d}")
            .set_properties(**{'width': '40px'})
            .set_table_styles([
                {'selector': 'table', 'props': [('width', '100%')]},
            ]),
        use_container_width=True
    )
    
    # IPIP 테스트 실행
    for i, persona in enumerate(test_personas):
        all_ipip_scores = []
        for j in range(0, 300, 50):
            batch_questions = ipip_questions['items'][j:j+50]
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
                        .background_gradient(cmap='YlOrRd')
                        .format("{:d}")
                        .set_properties(**{'width': '40px'})
                        .set_table_styles([
                            {'selector': 'table', 'props': [('width', '100%')]},
                        ]),
                    use_container_width=True
                )
    
    st.write("IPIP Test 완료")
    
    # BFI 테스트 실행 - 배치 크기를 더 작게 설정
    for i, persona in enumerate(test_personas):
        for j in range(0, 44, 10):  # 10개씩 배치 처리
            end_idx = min(j + 10, 44)  # 마지막 배치는 4개만 처리
            batch_questions = bfi_questions[j:end_idx]
            bfi_responses = get_llm_response(persona, batch_questions, 'BFI')
            
            if bfi_responses and 'responses' in bfi_responses:
                try:
                    scores = [r['score'] for r in bfi_responses['responses']]
                    current_scores = bfi_df.iloc[i].copy()
                    current_scores[j:j+len(scores)] = scores
                    bfi_df.iloc[i] = current_scores
                    bfi_df.loc['Average'] = bfi_df.iloc[:-1].mean()
                    
                    bfi_df_full.iloc[i] = current_scores
                    bfi_df_full.loc['Average'] = bfi_df_full.iloc[:-1].mean()
                    
                    # 진행률 계산 수정
                    total_questions = len(test_personas) * 44
                    current_progress = (i * 44 + end_idx) / total_questions
                    bfi_progress.progress(current_progress)
                    
                    bfi_table.dataframe(
                        bfi_df.fillna(0).round().astype(int).style
                            .background_gradient(cmap='YlOrRd')
                            .format("{:d}")
                            .set_properties(**{'width': '40px'})
                            .set_table_styles([
                                {'selector': 'table', 'props': [('width', '100%')]},
                            ]),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"BFI 점수 처리 중 오류: {str(e)}")
    
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