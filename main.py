from crewai.tools import BaseTool
import warnings
import os
from google.colab import userdata
from crewai import LLM
import smtplib
import yagmail


warnings.filterwarnings('ignore')

from crewai.tools import BaseTool

class NewsScraperTool(BaseTool):
    name: str = "NewsScraperTool"
    description: str = "Scrapes news data from fitness websites"

    def _run(self, query: str) -> str:  # Notice the parameter is just 'query: str'
        import requests
        from bs4 import BeautifulSoup
        import csv

        # URL of the Healthline Health News page
        url = "https://www.healthline.com/health-news"

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Data storage
            data = []

            # Extracting health news articles
            articles = soup.find_all("li", class_="css-18vzruc")

            # Loop through each article to extract details
            for article in articles:
                try:
                    # Extract title
                    title_tag = article.find("h2", class_="css-1rjem4a")
                    title = title_tag.text.strip() if title_tag else "No title"

                    # Extract publication date
                    date_tag = article.find("div", class_="css-5ry8xk")
                    publication_date = date_tag.text.strip() if date_tag else "No date"

                    # Extract description
                    description_tag = article.find("p", class_="css-ur5q1p")
                    description = description_tag.text.strip() if description_tag else "No description"

                    # Extract link
                    link_tag = article.find("a", class_="css-1wivj18")
                    link = link_tag["href"] if link_tag else "No link"

                    # Combine the extracted data
                    data.append([title, publication_date, description, link])

                except Exception as e:
                    print(f"Error processing an article: {e}")

            # Save the data to a CSV file
            csv_file = "healthline_health_news.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Title", "Publication Date", "Description", "Link"])
                writer.writerows(data)

            return csv_file
        except Exception as e:
            return f"Error scraping data: {str(e)}"

news_scraper_tool = NewsScraperTool()

class CSVReaderTool(BaseTool):
  name: str = "CSVReaderTool"
  description: str = "Reads CSV files"

  def _run(self, path: str) -> str:
    import pandas as pd

    fitness_df = pd.read_csv("healthline_health_news.csv")

    return fitness_df


csv_reader_tool = CSVReaderTool()

############     Under construction     ############

class EmailSenderTool(BaseTool):
  name: str = "EmailSenderTool"
  description: str = "Sends emails"

  def _run(self, email_add: str) -> str:

    import smtplib




#os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
#os.environ['GEMINI_API_KEY'] = 'AIzaSyB0b_73xWChdCrmTtpPen1DzQ3vvFEoDL8'
#g_api_key = os.environ['GOOGLE_API_KEY']


os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')


llm = LLM(model="gpt-4")
#llm = LLM(model="gemini/gemini-1.5-pro")

from crewai import Agent, Task, Crew

data_scraping_agent = Agent(
    role="Data Scraping Agent",
    goal="Scrape news data from fitness websites, save data to csv file and return the csv file name",
    backstory="""You are a professional Data Scraping Engineer with years of experience in scraping data from different websites.
                 Your task is to scrape news articles,
                 save then into a csv file and
                 pass the csv file name to the csv_reader_agent using the news_scraper_tool.""",
    llm=llm,  # Use the properly configured LLM
    tools=[news_scraper_tool],
    allow_delegation=False,
    verbose=True
)

csv_reader_agent = Agent(
    role="CSV File Reader Agent",
    goal="Read CSV File Data and return the data.",
    backstory="""You are a CSV File expert.
                 Your goal is to read the given csv file(by data scraping agent),
                 pass the data as a pandas dataframe using the csv_reader_tool to the newsletter_generator_agent.""",
    llm=llm,
    tools=[csv_reader_tool],
    allow_delegation=False,
    verbose=True
)

newsletter_generator_agent = Agent(
    role="Newsletter Generator Agent",
    goal="Generate health-related newsletters from using the data.",
    backstory="""You are an expert in generating newsletters.
                 Take the data passed by the csv_reader_agent and,
                 generate health-related newsletters using them.
                 Showcase the final newsletters to the user.""",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

data_scraping_task = Task(
    description=(
        "1. Scrape news data from fitness websites using the news_scraper_tool."
        "2. Save the scraped data into a CSV file with appropriate formatting."
        "3. Return the CSV file name to be used by the next agent."
    ),
    expected_output=(
        "A CSV file containing the scraped news data from fitness websites."
    ),
    agent=data_scraping_agent,
    human_input=False
)



csv_reading_task = Task(
    description=(
        "1. Read the data from the provided CSV file using the csv_reader_tool."
        "2. Convert the data into a pandas dataframe for easy manipulation and processing."
        "3. Pass the dataframe to the newsletter_generator_agent for further use."
    ),
    expected_output=(
        "A pandas dataframe containing the data from the CSV file."
    ),
    agent=csv_reader_agent,
    human_input=False
)



newsletter_generation_task = Task(
    description=(
        "1. Take the data passed by the csv_reader_agent."
        "2. Generate health-related newsletters based on the data."
        "3. Showcase the final newsletters to the user in an engaging and informative format."
    ),
    expected_output=(
        "A set of health-related newsletters generated from the provided data, ready to be showcased to the user."
    ),
    agent=newsletter_generator_agent,
    human_input=False
)


#email_sending_task =

newsletter_crew = Crew(
    agents=[data_scraping_agent, csv_reader_agent, newsletter_generator_agent],
    tasks=[data_scraping_task, csv_reading_task, newsletter_generation_task],
    verbose=True,
    process_type="sequential"  # Add explicit process type
)

user_query = input("Enter your business problem here:")
result = newsletter_crew.kickoff(inputs={"user_query": user_query})


# creates SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)
# start TLS for security
s.starttls()
# Authentication
s.login("kainatraisa@gmail.com", "#Iamborntowin21")
# message to be sent
message = """Newsletter 3:
Title: Carbonated Water May Promote Weight Loss, but There's a Catch
Publication Date: January 22, 2025
Description: New research suggests that drinking sparkling water could aid in weight loss. However, it's not as straightforward as it seems. Learn more about the benefits and potential risks. [Read More](https://www.healthline.com/health-news/could-sparkling-water-help-you-lose-weight-study-says-yes-with-catch)"""
# sending the mail
s.sendmail("kainatraisa@gmail.com", "kraisahossain@gmail.com", message)
# terminating the session
s.quit()





yag = yagmail.SMTP("kainatraisa@gmail.com")
yag.send(
    to="kraisahossain@gmail.com",
    subject="Automated Email",
    contents="Hello, this is an automated email from Python!",
)
print("Email sent successfully!")