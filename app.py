import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
import anthropic
import google.generativeai as genai

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
llm_choice = st.radio("LLM 선택", ("GPT-4", "Claude 3", "Gemini Pro"))

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
            # JSON 문자열 추출 및 정제
            content = response.choices[0].message.content
            # 코드 블록이 있는 경우 제거
            if content.startswith('```') and content.endswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            return json.loads(content.strip())
        
        elif llm_choice == "Claude 3":
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0
            )
            return json.loads(response.content[0].text)
        
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
                # JSON 형식이 아닌 경우 응답에서 JSON 부분만 추출 시도
                import re
                json_str = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_str:
                    return json.loads(json_str.group())
                raise Exception("Invalid JSON response")
            
    except Exception as e:
        st.error(f"LLM API 오류: {str(e)}")
        st.write("프롬프트:", prompt)
        st.write("응답:", response if 'response' in locals() else "No response")
        return None

# 테스트 실행 버튼
if st.button("테스트 시작"):
    all_results = {}
    
    # 빈 데이터프레임 초기화
    ipip_df = pd.DataFrame(
        np.nan, 
        index=[f"Persona {i+1}" for i in range(len(personas))] + ['Average'],
        columns=[f"Q{i+1}" for i in range(300)]
    )
    
    bfi_df = pd.DataFrame(
        np.nan, 
        index=[f"Persona {i+1}" for i in range(len(personas))] + ['Average'],
        columns=[f"Q{i+1}" for i in range(44)]
    )
    
    # CSV 저장용 데이터프레임 (소수점 유지)
    ipip_df_full = ipip_df.copy()
    bfi_df_full = bfi_df.copy()
    
    # 세로로 배치
    st.write("### IPIP Test 결과")
    ipip_table = st.empty()
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
    
    st.write("### BFI Test 결과")
    bfi_table = st.empty()
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
    
    for i, persona in enumerate(personas):
        # IPIP 테스트 (300개 질문을 50개씩 나누어 처리)
        all_ipip_scores = []
        for j in range(0, 300, 50):
            batch_questions = ipip_questions['items'][j:j+50]
            ipip_responses = get_llm_response(persona, batch_questions, 'IPIP')
            if ipip_responses and 'responses' in ipip_responses:
                scores = [r['score'] for r in ipip_responses['responses']]
                all_ipip_scores.extend(scores)
                
                # 부분 결과 업데이트
                current_scores = ipip_df.iloc[i].copy()
                current_scores[j:j+len(scores)] = scores
                ipip_df.iloc[i] = current_scores
                ipip_df.loc['Average'] = ipip_df.iloc[:-1].mean()
                
                # CSV용 데이터프레임도 업데이트
                ipip_df_full.iloc[i] = current_scores
                ipip_df_full.loc['Average'] = ipip_df_full.iloc[:-1].mean()
                
                # 로그 스케일 적용 및 화면 업데이트
                ipip_df_log = np.log1p(ipip_df) / np.log1p(5) * 5
                ipip_table.dataframe(
                    ipip_df_log.style
                        .background_gradient(cmap='YlOrRd')
                        .format("{:.0f}")  # 정수로 표시
                        .set_properties(**{'width': '50px'})
                )
        
        # BFI 테스트
        bfi_responses = get_llm_response(persona, bfi_questions[:44], 'BFI')
        if bfi_responses and 'responses' in bfi_responses:
            try:
                scores = [r['score'] for r in bfi_responses['responses']]
                if len(scores) == len(bfi_df.columns):
                    # 화면 표시용 데이터프레임 업데이트
                    bfi_df.iloc[i] = scores
                    bfi_df.loc['Average'] = bfi_df.iloc[:-1].mean()
                    
                    # CSV 저장용 데이터프레임 업데이트
                    bfi_df_full.iloc[i] = scores
                    bfi_df_full.loc['Average'] = bfi_df_full.iloc[:-1].mean()
                    
                    # 로그 스케일 적용 및 화면 업데이트
                    bfi_df_log = np.log1p(bfi_df) / np.log1p(5) * 5
                    bfi_table.dataframe(
                        bfi_df_log.style
                            .background_gradient(cmap='YlOrRd')
                            .format("{:.0f}")  # 정수로 표시
                            .set_properties(**{'width': '50px'})
                    )
            except Exception as e:
                st.error(f"BFI 점수 처리 중 오류: {str(e)}")
        
        # 결과 저장
        all_results[f"persona_{i+1}"] = {
            "personality": persona['personality'],
            "IPIP_responses": {"responses": [{"score": s} for s in all_ipip_scores]} if all_ipip_scores else None,
            "BFI_responses": bfi_responses
        }
    
    # CSV 파일 생성 (소수점 10자리까지 유지)
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