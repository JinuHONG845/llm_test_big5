import streamlit as st
import pandas as pd
import json
import openai
import anthropic
import google.generativeai as genai

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
    else:
        question_list = [q['question'] for q in questions]
    
    # 프롬프트 구성
    prompt = f"""Based on this persona: {', '.join(persona['personality'])}

For each question, provide a rating from 1-5 where:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree

Return ONLY a JSON object in this exact format:
{{
    "responses": [
        {{"question": "<question text>", "score": <1-5>}},
        ...
    ]
}}

Questions to rate:
{json.dumps(question_list[:50], indent=2)}"""  # 한 번에 50개 질문만 처리
    
    try:
        if llm_choice == "GPT-4":
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content)
        
        elif llm_choice == "Claude 3":
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return json.loads(response.content[0].text)
        
        else:  # Gemini Pro
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7
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
    progress_bar = st.progress(0)
    
    for i, persona in enumerate(personas):
        st.write(f"### 페르소나 {i+1}/50 처리 중...")
        
        # IPIP 테스트
        ipip_responses = get_llm_response(persona, ipip_questions['items'], 'IPIP')
        
        # BFI 테스트
        bfi_responses = get_llm_response(persona, bfi_questions, 'BFI')
        
        # 결과 저장
        all_results[f"persona_{i+1}"] = {
            "personality": persona['personality'],
            "IPIP_responses": ipip_responses,
            "BFI_responses": bfi_responses
        }
        
        # 진행률 업데이트
        progress_bar.progress((i + 1) / len(personas))
    
    # 결과를 CSV로 변환
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    
    # 결과 다운로드 버튼
    st.download_button(
        label="결과 다운로드 (CSV)",
        data=results_df.to_csv(index=True),
        file_name="personality_test_results.csv",
        mime="text/csv"
    )
    
    st.success("모든 테스트가 완료되었습니다!")