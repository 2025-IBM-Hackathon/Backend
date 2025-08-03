# app.py
# app.py

import streamlit as st
from backend import classify_message

st.set_page_config(page_title="스미싱 가디언", page_icon="📭")

st.title("📭 스미싱 가디언")
st.markdown("문자 메시지를 입력하면 스미싱 여부를 판단해드립니다.")

# 사용자 입력 받기
user_input = st.chat_input("여기에 수신한 메시지를 입력하세요")

# 이전 판단 결과 저장
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# 메시지 입력 시 처리
if user_input:
    with st.spinner("메시지를 분석 중입니다..."):
        result = classify_message(user_input)

    # 결과 저장
    st.session_state.last_result = result

    # 사용자 메시지 출력
    st.chat_message("user").write(user_input)

    # confidence 값에서 '%' 제거 후 float 변환
    confidence_str = result['confidence'].replace('%', '').strip()
    confidence_value = float(confidence_str)

    # UI 출력
    st.chat_message("ai").markdown(f"""
    ### ✅ 판단 결과
    **최종 판단:** `{result['label']}`  
    **위험도:** `{confidence_value}%`  
    **판단 근거:**  
    {result['reason']}
    """)

    # 추가 안내: 스미싱인 경우
    if result['label'] == "스미싱":
        with st.expander("📢 대응 가이드 보기"):
            st.markdown("""
            - **문자에 포함된 링크를 클릭하지 마세요.**
            - **의심스러운 번호로 전화하지 마세요.**
            - 이미 클릭했다면 **금융 앱, 공인인증서 등을 삭제**하고,
              가까운 경찰서 또는 금융감독원에 신고하세요.
            - **보호자에게 알리기** 버튼을 누르면 가족/지인에게 자동으로 위험 메시지를 전송할 수 있습니다.
            """)
            st.button("📨 보호자에게 알리기", disabled=True)  # 기능 구현 전에는 비활성화

        with st.expander("🔍 피해 여부 확인하기"):
            st.markdown("""
            이미 클릭했거나, 앱 설치 등이 의심된다면 아래를 확인해보세요.
            - 휴대폰에 알 수 없는 앱이 설치되어 있지 않나요?
            - 통신사, 은행으로부터 비정상적인 알림을 받지 않았나요?
            - 의심 문자 수신 후 **정보 유출 알림**을 받은 적 있나요?
            """)
            st.button("피해 점검 도구 연결 (준비 중)", disabled=True)

