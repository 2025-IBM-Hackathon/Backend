import smtplib
from email.message import EmailMessage

def send_alert_email(user_name, to_email, message_text):
    # Gmail SMTP 서버 정보
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # 보내는 사람 이메일 정보
    from_email = "zenshim70@gmail.com"
    from_password = "alhd wljg kfmo ekmt"  # 앱 비밀번호 사용

    # 이메일 메시지 작성
    msg = EmailMessage()
    msg["Subject"] = "⚠️ 스미싱 의심 문자 감지 알림"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(f"스미싱으로 의심되는 문자가 {user_name}님께 도착했습니다:\n\n{message_text}")

    try:
        # SMTP 서버에 연결
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # TLS로 보안 연결
            server.login(from_email, from_password)
            server.send_message(msg)
        print("📧 보호자에게 이메일 전송 완료")
    except Exception as e:
        print(f"이메일 전송 실패: {e}")



if __name__ == '__main__':
    send_alert_email('김이름','zenshim71@gmail.com', 'This is a test message body.')