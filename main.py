from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
import os
import smtplib
import ssl
from email.message import EmailMessage
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
from dotenv import load_dotenv
from litellm import completion
from langchain_community.chat_models import ChatLiteLLM

# Load environment variables
load_dotenv()

# Configure LiteLLM (automatically uses GEMINI_API_KEY from environment)
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

class NewsScraperTool(BaseTool):
    name: str = "NewsScraperTool"
    description: str = "Scrapes news data from fitness websites and saves to CSV"

    def _run(self, query: str) -> str:
        try:
            url = "https://www.healthline.com/health-news"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            data = []

            articles = soup.find_all("li", class_="css-18vzruc")

            for article in articles:
                try:
                    title = article.find("h2", class_="css-1rjem4a").text.strip()
                    date = article.find("div", class_="css-5ry8xk").text.strip()
                    description = article.find("p", class_="css-ur5q1p").text.strip()
                    link = article.find("a", class_="css-1wivj18")["href"]
                    data.append([title, date, description, link])
                except Exception as e:
                    continue

            csv_file = "health_news.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Title", "Date", "Description", "Link"])
                writer.writerows(data)

            return csv_file
        except Exception as e:
            return f"Error: {str(e)}"

class CSVReaderTool(BaseTool):
    name: str = "CSVReaderTool"
    description: str = "Reads and processes CSV files"

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            return df.to_markdown(index=False)
        except Exception as e:
            return f"Error reading CSV: {str(e)}"

class NewsletterGeneratorTool(BaseTool):
    name: str = "NewsletterGeneratorTool"
    description: str = "Generates newsletters using Gemini AI"

    def _run(self, data: str) -> str:
        try:
            prompt = f"""
            Create a professional health newsletter using the following articles:
            {data}

            Include these elements:
            1. Engaging introduction
            2. 3-5 key highlights with summaries
            3. Expert analysis section
            4. Links to original articles
            5. Closing recommendations

            Format in clean HTML with proper headings and sections.
            """

            response = completion(
                model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating newsletter: {str(e)}"

class EmailSenderTool(BaseTool):
    name: str = "EmailSenderTool"
    description: str = "Sends emails using Gmail"
    default_recipient: str = "fliptechy09@gmail.com"

    def _run(self, recipient: str = None, subject: str = "Health Newsletter", body: str = "") -> str:
        try:
            # Debug prints to verify inputs
            print(f"\n🔧 DEBUG TOOL INPUTS 🔧")
            print(f"Received recipient: {recipient}")
            print(f"Default recipient: {self.default_recipient}")
            
            sender_email = os.getenv("GMAIL_USER")
            recipient_email = self.default_recipient
            if not isinstance(recipient_email, str) or "@" not in recipient_email:
                recipient_email = self.default_recipient
                print(f"⚠️ Invalid recipient, using default: {recipient_email}")
            print(f"DEBUG: Final recipient - {recipient_email}")
            password = os.getenv("GMAIL_PASSWORD")
            
            # Validate email format
            if "@" not in recipient_email or "." not in recipient_email.split("@")[1]:
                return f"Invalid email address: {recipient_email}"

            # Create email message
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg.set_content(body, subtype='html')

            # Send email using SMTP
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.send_message(msg)
                
            return f"Email sent successfully to {recipient_email}"
        except Exception as e:
            return f"Error sending email: {str(e)}"


# Initialize tools
news_scraper = NewsScraperTool()
csv_reader = CSVReaderTool()
newsletter_generator = NewsletterGeneratorTool()
email_sender = EmailSenderTool()

# Configure Gemini LLM using LiteLLM
llm = ChatLiteLLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.5
)

# Create agents
data_scraping_agent = Agent(
    role="Data Scraping Specialist",
    goal="Scrape health news data and save to CSV",
    backstory="Expert web scraper with extensive experience in data extraction from health websites.",
    tools=[news_scraper],
    llm=llm,
    verbose=True,
)

csv_processing_agent = Agent(
    role="Data Processing Specialist",
    goal="Process and analyze CSV data",
    backstory="Skilled data analyst with expertise in data cleaning and preparation.",
    tools=[csv_reader],
    llm=llm,
    verbose=True,
)

newsletter_agent = Agent(
    role="Health Newsletter Editor",
    goal="Create engaging health newsletters using Gemini AI",
    backstory="AI-powered content creator specialized in medical journalism.",
    tools=[newsletter_generator],
    llm=llm,
    verbose=True,
)

email_agent = Agent(
    role="Email Communications Specialist",
    goal="Send formatted newsletters via email",
    backstory="A dedicated specialist focused on crafting and sending emails.",
    tools=[email_sender],
    llm=llm,
    verbose=True,
)

# Create tasks (remain unchanged from original)
scraping_task = Task(
    description="Scrape latest health news articles and save to CSV",
    expected_output="CSV file containing health news data",
    agent=data_scraping_agent,
    output_file="health_news.csv",
)

processing_task = Task(
    description="Process CSV data into readable format for newsletter creation",
    expected_output="Cleaned and formatted news data in markdown format",
    agent=csv_processing_agent,
    context=[scraping_task],
)

newsletter_task = Task(
    description="Create engaging newsletter from health news data using Gemini AI",
    expected_output="Well-formatted newsletter HTML content with medical insights",
    agent=newsletter_agent,
    context=[processing_task],
)

email_task = Task(
    description="Send newsletter via email to specified recipient",
    expected_output="Confirmation of email delivery",
    agent=email_agent,
    context=[newsletter_task],
    inputs={
        "recipient": "fliptechy09@gmail.com",
        "subject": "Newsletter of the day",
        "body": newsletter_task.output
    }
)

# Create and run crew (remain unchanged from original)
newsletter_crew = Crew(
    agents=[data_scraping_agent, csv_processing_agent, newsletter_agent, email_agent],
    tasks=[scraping_task, processing_task, newsletter_task, email_task],
    verbose=True,
)

if __name__ == "__main__":
    recipient_email = "fliptechy09@gmail.com"
    newsletter_subject = "Newsletter of the day"
    # Add this before the kickoff call
    if "@" not in recipient_email or "." not in recipient_email.split("@")[1]:
        raise ValueError(f"Invalid recipient email: {recipient_email}")
    
    print(f"\n🚀 STARTING PROCESS WITH:")
    print(f"Recipient: {recipient_email}")
    print(f"Subject: {newsletter_subject}")

    result = newsletter_crew.kickoff(
        inputs={
            "recipient": "fliptechy09@gmail.com",
            "subject": newsletter_subject,
            "body": newsletter_task.output  # Explicitly pass newsletter content
        }
    )

    print(f"\n\nProcess completed! Mail sent to {recipient_email}.")
    print(f"\n📨 FINAL RESULT:")
    print(result)