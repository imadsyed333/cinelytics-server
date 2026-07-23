import requests
from dotenv import load_dotenv
import os
from models import MovieData, MovieReview, Genre, ProductionCompany
from langchain.tools import tool

load_dotenv()

system_prompt = """You are a film industry analyst specializing in box office performance.

## Capabilities

- `fetch_movie_data`: loads a JSON of movie data into the conversation.
- `fetch_reviews`: loads a list of strings representing movie reviews into the conversation.
- `describe_performance`: Describes the box office performance of a movie based on its budget and revenue.

Your task is to analyze movie box office performance by combining quantitative metrics with audience/critic sentiment. Utilize as much data available as possible to provide a comprehensive and insightful analysis.

## Required Process

1. ALWAYS fetch movie data first using `fetch_movie_data`
2. ALWAYS fetch reviews using `fetch_reviews` 
3. Use `describe_performance` with the budget and revenue to categorize box office success
4. Analyze the sentiment and themes from reviews to understand audience reception
5. Correlate audience reception with box office performance

Follow these rules strictly:
- You MUST use ALL tools for every analysis
- Base your reasoning ONLY on the provided data
- Do NOT invent facts or external knowledge
- Reviews provide critical insight into word-of-mouth, audience satisfaction, and long-term performance
- Be concise but insightful
- Focus on causal factors (why performance happened)
- Avoid vague statements like "it depends" or "various factors"""

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

@tool
def describe_performance(revenue: int, budget: int) -> str:
    """
    Categorizes box office performance based on revenue-to-budget ratio.
    Returns: 'was a hit' (3x+), 'was a moderate success' (2-3x), 'broke-even' (1.5-2x), or 'underperformed' (<1.5x).
    
    Args:
        revenue: Total worldwide box office revenue in US dollars
        budget: Production budget in US dollars
    """
    if budget == 0:
        return "Unknown"
    
    ratio = revenue / budget
    if ratio >= 3.0:
        return "was a hit"
    elif ratio >= 2.0:
        return "was a moderate success"
    elif ratio >= 1.5:
        return "broke-even"
    else:
        return "underperformed"

@tool
def fetch_movie_data(movie_id: int) -> MovieData:
    """
    Fetches comprehensive movie data including budget, revenue, ratings, genres, and production companies.
    Foundation for box office analysis. ALWAYS call this first to establish baseline metrics.
    
    Args:
        movie_id: The unique identifier for the movie (e.g., TMDB movie ID)
    """
    response = requests.get(f"{API_URL}/movie/{movie_id}", headers={"Authorization": f"Bearer {API_KEY}"})

    print("Fetching movie data...")

    if response.status_code != 200:
        raise Exception(f"Failed to fetch movie data: {response.status_code} - {response.text}")
    
    data = response.json()
    return parse_movie_data(data)

def parse_genre(data: dict) -> Genre:
    return Genre(
        name=data.get("name", "")
    )

def parse_production_company(data: dict) -> ProductionCompany:
    return ProductionCompany(
        name=data.get("name", ""),
        origin_country=data.get("origin_country", "")
    )

def parse_movie_data(data: dict) -> MovieData:
    return MovieData(
        title=data.get("title", ""),
        release_date=data.get("release_date", ""),
        budget=data.get("budget", 0),
        rating=data.get("vote_average", 0.0),
        revenue=data.get("revenue", 0),
        overview=data.get("overview", ""),
    )

@tool
def fetch_reviews(movie_id: int) -> list[MovieReview]:
    """
    Fetches audience and critic reviews for qualitative sentiment analysis.
    Reveals audience satisfaction, word-of-mouth factors, and specific strengths/weaknesses.
    ALWAYS call after fetch_movie_data to understand the 'why' behind performance.
    
    Args:
        movie_id: The unique identifier for the movie (e.g., TMDB movie ID)
    """
    response = requests.get(f"{API_URL}/movie/{movie_id}/reviews", headers={"Authorization": f"Bearer {API_KEY}"})

    print("Fetching reviews...")
    if response.status_code != 200:
        raise Exception(f"Failed to fetch reviews: {response.status_code} - {response.text}")
    data = response.json()
    return parse_reviews(data.get("results", []))

def parse_reviews(data: list) -> list[MovieReview]:
    reviews = []
    for item in data:
        reviews.append(MovieReview(
            id=item.get("id", ""),
            author=item.get("author", ""),
            content=item.get("content", ""),
        ))
    return reviews[:min(5, len(reviews))]

def stringify_reviews(reviews: list[MovieReview]) -> str:
    review_str = ""
    for review in reviews[:min(5, len(reviews))]:  # Limit to first 5 reviews for brevity
        review_str += f"Review by {review.author}:\n{review.content}\n\n"
    return review_str.strip()