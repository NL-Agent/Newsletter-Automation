import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()
email_user = os.getenv("GMAIL_USER")
email_password = os.getenv("GMAIL_PASSWORD")

with open('config.json', 'r') as file:
    config = json.load(file)
email = config['newsletter']['email_to']
prompt_message = config['newsletter']['what_you_need']

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Email details from config
smtp_server = "smtp.gmail.com"  
smtp_port = 587  
sender_email = email_user 
sender_password = email_password  
recipient_email = email

def send_newsletter_via_email(subject: str, body: str):
    """
    Sends the generated newsletter as an email to the recipient.
    """
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  
        server.login(sender_email, sender_password)
        text = msg.as_string()

        server.sendmail(sender_email, recipient_email, text)
        server.quit()

        print(f"Newsletter sent to {recipient_email}")

    except Exception as e:
        print(f"Error sending email: {str(e)}")

def news_scraper_tool() -> pd.DataFrame:
    """
    Scrapes health news from Healthline and returns a DataFrame with the data.
    """
    url = "https://www.healthline.com/health-news"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    data = []
    articles = soup.find_all("li", class_="css-yah9nt")

    if not articles:
        print("No articles found on the page.")
        return pd.DataFrame()

    for article in articles:
        # Extract Title
        title_tag = article.find("h2", class_="css-16o4j9x")
        title = title_tag.text.strip() if title_tag else "No title"

        # Extract Publication Date
        date_tag = article.find("div", class_="css-5ry8xk")
        publication_date = date_tag.text.strip() if date_tag else "No date"

        # Extract Description
        description_tag = article.find("p", class_="css-1hw29i9")
        description = description_tag.text.strip() if description_tag else "No description"

        # Extract Link
        link_tag = article.find("a", class_="css-a63gyd")
        link = link_tag["href"] if link_tag else "No link"
        if not link.startswith("http"):
            link = f"https://www.healthline.com{link}"

        # Extract Image URL
        image_tag = article.find("lazy-image")
        image_url = image_tag["src"] if image_tag else "No image"

        data.append([title, publication_date, description, link, image_url])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Title", "Publication Date", "Description", "Link", "Image URL"])

    return df



llm_with_tools = llm.bind_tools([news_scraper_tool])

single_content = True  # Need this to move to config file

format_instructions = (
    "🔶 YOU ARE IN LIST CONTENT MODE (single_content=False):\n"
    "1. Create 3-5 news items using:\n"
    "   - <h3> for item headings\n"
    "   - <ul> with <li> bullet points\n"
    "   - Horizontal dividers between items\n"
    "2. Each item contains:\n"
    "   a. Concise 2-sentence summary\n"
    "   b. 3 key points\n"
    "   c. 1 relevant link\n"
    "3. Prioritize scannability with clear hierarchy"
) if not single_content else (
    "🔷 YOU ARE IN SINGLE STORY MODE (single_content=True):\n"
    "1. Create ONE in-depth story using:\n"
    "   - <h2> for main headlines\n"
    "   - <p> paragraphs with <span> highlights\n"
    "   - Embedded links as <a href='...'>clean anchors</a>\n"
    "2. Structure with:\n"
    "   a. Contextual background\n"
    "   b. Expert analysis\n"
    "   c. Visual data representations\n"
    "   d. Future implications"
    "3. Don't use list:\n"
    "   a. <ul> or <li> tag donn't use\n"
    "   b. Generate in paragraph type, one single paragraph\n"
    "   c. Don't make multiple headings and multiple paragraph or spans or images\n"
    "4. Your work:\n"
    "   a. Get the best match context and Make a Heading for that\n"
    "   b. Get its content and Make a single nice formatted paragraph for that\n"
)

sys_msg = SystemMessage(
    content=f"""
    You are a professional newsletter architect. Current mode: {'Single Story' if single_content else 'List Format'}
    
    {format_instructions}
    
    GENERAL RULES:
    • Never show raw URLs - always use <a> tags
    • Maintain consistent styling with CSS classes
    • Balance text/media ratio (30% visual elements)
    • Include 1 primary CTA in footer
    • Verify all links are HTTPS
    • Mobile-optimized layout"""
)



def newsletter_assistant(state: MessagesState):
    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("newsletter_assistant", newsletter_assistant)
builder.add_node("tools", ToolNode([news_scraper_tool]))

builder.add_edge(START, "newsletter_assistant") 

builder.add_conditional_edges("newsletter_assistant", tools_condition)
builder.add_edge("tools", "newsletter_assistant")


graph = builder.compile()

messages = [HumanMessage(content=prompt_message)]
messages = graph.invoke({"messages": messages})

if messages["messages"]:
    last_message = messages["messages"][-1]
    newsletter_content = last_message.content
else:
    newsletter_content = "No newsletter content generated."

formatted_newsletter = f"""
<html>
  <head>
    <style>
      body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
      h1 {{ color: #2c3e50; }}
      ul {{ list-style-type: none; padding: 0; }}
      li {{ margin-bottom: 15px; padding: 10px; border-left: 3px solid #3498db; }}
      a {{ color: #3498db; text-decoration: none; }}
      .container {{ max-width: 600px; margin: 0 auto; }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>📬 Your Daily Health Brief</h1>
      {newsletter_content}
      <p style="color: #7f8c8d; margin-top: 30px;">Stay informed, stay healthy!<br>The Health Update Team</p>
    </div>
  </body>
</html>
"""

subject = "🌡️ Your Latest Health News Update"
send_newsletter_via_email(subject, formatted_newsletter)
