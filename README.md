# Cinelytics Server

Cinelytics Server is a FastAPI-based backend service that leverages advanced language models to analyze and explain the box office performance of movies. It integrates with the HuggingFace Transformers library and LangChain to provide insightful, data-driven explanations for why a movie succeeded or failed financially, based on real movie data.

## Features

- **Movie Data Fetching:** Retrieves movie data (title, release date, budget, revenue, overview) from the TMDB API.
- **Performance Categorization:** Classifies movies as "Hit", "Moderate Success", "Break-even", or "Underperformed" based on revenue-to-budget ratio.
- **LLM-Powered Analysis:** Uses a quantized Microsoft Phi-3 model (via HuggingFace and LangChain) to generate specific, non-generic explanations for a movie's performance.
- **REST API Endpoint:** Exposes an `/analyze/{movie_id}` endpoint that returns an LLM-generated analysis for a given movie.

## How It Works

1. **Request:** Client sends a GET request to `/analyze/{movie_id}`.
2. **Data Fetch:** The server fetches movie data from an external API using the provided `movie_id`.
3. **Performance Evaluation:** The server categorizes the movie's performance based on its budget and revenue.
4. **LLM Analysis:** The movie data and performance category are passed to a language model, which returns a detailed, data-driven explanation.
5. **Response:** The server returns the analysis as a JSON response.

## Project Structure

- `main.py` — FastAPI app, LLM pipeline, and API endpoint.
- `models.py` — Pydantic models for movie data.
- `utils.py` — Utility functions for fetching movie data, performance description, and system prompt.

## Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd cinelytics-server
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set environment variables:**
   - Create a `.env` file with your API credentials:
     ```env
     API_KEY=your_api_key
     API_URL=https://api.themoviedb.org/3
     ```
5. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```
6. **Test the API:**
   - Visit: `http://localhost:8000/analyze/<movie_id>`

## Requirements

- Python 3.8+
- FastAPI
- Transformers
- LangChain
- torch
- python-dotenv
- requests
- Pydantic

## Example

```json
{
  "analysis": "The movie 'Example Title' released in 2023 had a budget of $50M and a revenue of $120M. Its strong box office performance can be attributed to..."
}
```
