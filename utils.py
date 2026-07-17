import requests
from dotenv import load_dotenv
import os
from models import MovieData, MovieReview

load_dotenv()

system_prompt = """You are a film industry analyst specializing in box office performance.

## Capabilities

- `fetch_movie_data`: loads a JSON of movie data into the conversation.
- `fetch_reviews`: loads a list of strings representing movie reviews into the conversation.

Your task is to analyze movie box office performance by combining quantitative metrics with audience/critic sentiment.

## Required Process

1. ALWAYS fetch movie data first using `fetch_movie_data`
2. ALWAYS fetch reviews using `fetch_reviews` 
3. Analyze the sentiment and themes from reviews to understand audience reception
4. Correlate audience reception with box office performance

Follow these rules strictly:
- You MUST use BOTH tools for every analysis
- Base your reasoning ONLY on the provided data
- Do NOT invent facts or external knowledge
- Reviews provide critical insight into word-of-mouth, audience satisfaction, and long-term performance
- Be concise but insightful
- Focus on causal factors (why performance happened)
- Avoid vague statements like "it depends" or "various factors"

Structure your response exactly as follows:

1. Performance Summary (1–2 sentences with revenue/budget figures)
2. Audience Reception (2-3 sentences summarizing review sentiment and key themes)
3. Key Factors (bullet points, 3–6 items - must include at least one factor based on reviews)
4. Final Assessment (1–2 sentences with clear judgment)

Each factor must clearly explain cause → effect."""

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

def fetch_movie_data(movie_id: int) -> MovieData:
    """
    Fetch structured movie data regarding movie with id <movie_id>
    """
    response = requests.get(f"{API_URL}/movie/{movie_id}", headers={"Authorization": f"Bearer {API_KEY}"})

    print("Fetching movie data...")

    if response.status_code != 200:
        raise Exception(f"Failed to fetch movie data: {response.status_code} - {response.text}")
    
    data = response.json()
    return parse_movie_data(data)

def parse_movie_data(data: dict) -> MovieData:
    return MovieData(
        title=data.get("title", ""),
        release_date=data.get("release_date", ""),
        budget=data.get("budget", 0),
        rating=data.get("vote_average", 0.0),
        revenue=data.get("revenue", 0),
        overview=data.get("overview", ""),
    )

def fetch_reviews(movie_id: int) -> list[MovieReview]:
    """
    Fetch list of strings representing reviews for movie with id <movie_id>
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
    return reviews

def stringify_reviews(reviews: list[MovieReview]) -> str:
    review_str = ""
    for review in reviews[:min(5, len(reviews))]:  # Limit to first 5 reviews for brevity
        review_str += f"Review by {review.author}:\n{review.content}\n\n"
    return review_str.strip()

def describe_performance(revenue: int, budget: int) -> str:
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