# Email_Test.py
import os
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv

def send_test_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends a test email using Gmail SMTP server
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get credentials
        sender_email = os.getenv("GMAIL_USER")
        password = os.getenv("GMAIL_PASSWORD")
        
        # Validate credentials
        if not sender_email or not password:
            return "Error: Missing email credentials in environment variables"
            
        # Validate email format
        if "@" not in recipient or "." not in recipient.split("@")[1]:
            return f"Invalid email address: {recipient}"

        # Create email message
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient
        msg.set_content(body, subtype='html')

        # Send email using SMTP
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.send_message(msg)
            
        return f"Test email sent successfully to {recipient}"
        
    except Exception as e:
        return f"Error sending test email: {str(e)}"

if __name__ == "__main__":
    # Test configuration
    test_recipient = "kainatraisa@gmail.com"  # Replace with your test email
    test_subject = "[TEST] Health Newsletter - System Check"
    test_body = """
    <html>
        <body>
            <h1 style="color: #2e6c80;">This is a test email</h1>
            <p>If you're reading this, the email sending functionality is working correctly!</p>
            <ul>
                <li>✅ SMTP Connection Established</li>
                <li>✅ Authentication Successful</li>
                <li>✅ HTML Content Rendered</li>
            </ul>
            <p>System status: <strong style="color: green;">Operational</strong></p>
        </body>
    </html>
    """
    
    # Send test email
    print("🚀 Sending test email...")
    result = send_test_email(
        recipient=test_recipient,
        subject=test_subject,
        body=test_body
    )
    
    # Show results
    print("\n📨 Test Results:")
    print(result)
    print("\n⚠️ Note: Check spam folder if you don't see the email in your inbox.")