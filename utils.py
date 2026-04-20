import requests
from dotenv import load_dotenv
import os
from models import MovieData, MovieReview

load_dotenv()

system_prompt = """You are a film industry analyst with expertise in box office performance. You have the ability to take various data points about a movie and analyze how those factors contributed to the movie's financial success or failure. You are skilled at identifying specific reasons for a movie's performance based on its budget, revenue, release year, title, and overview. Your analysis is concise, insightful, and directly supported by the provided data. You avoid making generic statements and focus on specific factors that influenced the movie's success or failure."""

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

def fetch_movie_data(movie_id: int) -> MovieData:
    response = requests.get(f"{API_URL}/movie/{movie_id}", headers={"Authorization": f"Bearer {API_KEY}"})

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
    response = requests.get(f"{API_URL}/movie/{movie_id}/reviews", headers={"Authorization": f"Bearer {API_KEY}"})
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
        return "Hit"
    elif ratio >= 2.0:
        return "Moderate Success"
    elif ratio >= 1.5:
        return "Break-even"
    else:
        return "Underperformed"