from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
import time

from utils import fetch_movie_data, fetch_reviews, describe_performance, system_prompt
from models import AnalysisResponse

model = ChatOllama(model="gemma4:e2b", temperature=0, reasoning=False)
structured_model = model.with_structured_output(AnalysisResponse)

agent = create_agent(
    model=model,
    tools=[fetch_movie_data, fetch_reviews, describe_performance],
    system_prompt=system_prompt,
    response_format=ToolStrategy[AnalysisResponse](AnalysisResponse),
)

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
    

    start_time = time.time()
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Provide a detailed box office performance analysis for the movie with movie ID:{movie_id}"
        }]
    })
    end_time = time.time()
    
    print("Time taken for analysis:", end_time - start_time, "seconds")
    print(result.keys())
    return result['structured_response']