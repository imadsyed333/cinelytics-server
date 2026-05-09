from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_ollama import ChatOllama
import time

from models import AnalysisResponse
from utils import describe_performance, fetch_movie_data, fetch_reviews, stringify_reviews, system_prompt

model = ChatOllama(model="qwen3.5:4b", temperature=0, validate_model_on_init=True, max_tokens=128, reasoning=False)

analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResponse)

review_prompt = PromptTemplate.from_template(
    "Here are some reviews for a movie:\n{reviews}\nI want you to provide me with a concise paragraph summarizing the overall sentiment and key points mentioned in these reviews."
)

analysis_prompt = PromptTemplate.from_template(
    "{system_prompt}\n"
    "The movie {title} ({release_date}) had a budget of ${budget}, generated ${revenue} in revenue, and received a rating of {rating}/10. Here's a brief overview of the movie: {overview}\n\nThe audience's sentiment based on reviews is: {sentiment}\n\nBased on this information, provide exactly three specific reasons to explain why the movie {performance} at the box office.\n\n"
    "Return ONLY valid JSON matching this schema. Do not return markdown, numbering, or extra text.\n"
    "{format_instructions}"
)

parser = StrOutputParser()

review_chain = review_prompt | model | parser

analysis_chain = analysis_prompt | model | analysis_parser

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/analyze/{movie_id}")
def analyze(movie_id: int):
    movie_data = fetch_movie_data(movie_id)

    movie_reviews = fetch_reviews(movie_id)
    reviews_str = stringify_reviews(movie_reviews)

    performance = describe_performance(movie_data.revenue, movie_data.budget)
    start_time = time.time()
    sentiment = review_chain.invoke({"reviews": reviews_str})
    end_time = time.time()
    
    print("Time taken for sentiment analysis:", end_time - start_time, "seconds")

    start_time = time.time()
    response = analysis_chain.invoke({
        "system_prompt": system_prompt,
        "title": movie_data.title,
        "release_date": movie_data.release_date,
        "budget": movie_data.budget,
        "revenue": movie_data.revenue,
        "rating": movie_data.rating,
        "overview": movie_data.overview,
        "performance": performance,
        "sentiment": sentiment,
        "format_instructions": analysis_parser.get_format_instructions(),
    })
    end_time = time.time()
    
    print("Time taken for analysis:", end_time - start_time, "seconds")
    return response
