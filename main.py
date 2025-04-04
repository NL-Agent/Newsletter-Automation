from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json


load_dotenv()

with open('config.json', 'r') as file:
    config = json.load(file)

email = config['newsletter']['email_to']
prompt_message = config['newsletter']['what_you_need']

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def news_scraper_tool() -> pd.DataFrame:
    """
    Scrapes health news from Healthline and returns a DataFrame with the data.
    """

    url = "https://www.healthline.com/health-news"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensures we stop on bad responses
    soup = BeautifulSoup(response.text, 'html.parser')

    data = []
    articles = soup.find_all("li", class_="css-yah9nt")  # Updated selector for articles

    if not articles:
        print("No articles found on the page.")
        return pd.DataFrame()  # Return an empty DataFrame if no articles are found

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

sys_msg = SystemMessage(
    content=(
        "You are an expert newsletter assistant capable of generating high-quality newsletters from real-time scraped news data. "
        "Your task is to process the latest news, summarize key points, and format them into engaging newsletter-style content. "
        "When a user requests specific topics (e.g., 'Give me 5 news about healthcare'), scrape relevant news, extract insights, "
        "and generate the requested number of newsletters in a concise and informative manner."
    )
)


def newsletter_assistant(state: MessagesState):
    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("newsletter_assistant", newsletter_assistant)
builder.add_node("tools", ToolNode([news_scraper_tool]))

builder.add_edge(START, "newsletter_assistant")  # Fix: Use string instead of function reference

builder.add_conditional_edges("newsletter_assistant", tools_condition)
builder.add_edge("tools", "newsletter_assistant")


graph = builder.compile()

messages = [HumanMessage(content=prompt_message)]
messages = graph.invoke({"messages": messages})


for m in messages["messages"]:
    m.pretty_print()
