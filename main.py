from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import describe_performance, fetch_movie_data, fetch_reviews, stringify_reviews, system_prompt

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": 0},
    quantization_config=bnb_config,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    return_full_text=False,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

review_prompt = PromptTemplate.from_template(
    "<|user|>\nHere are some reviews for a movie:\n{reviews}\nI want you to provide me with a concise summary of the overall sentiment and key points mentioned in these reviews.\n<|end|>\n"
    "<|assistant|>\n"
)

analysis_prompt = PromptTemplate.from_template(
    "<|system|>\n{system_prompt}<|end|>\n"
    "<|user|>\nThe movie {title} ({release_date}) has a budget of ${budget} and generated a revenue of ${revenue}. It has a rating of {rating}/10. A brief overview of the movie: {overview}\nThe audience's sentiment based on reviews is: {sentiment}\nThe movie's performance is categorized as: {performance}.\n\nBased on this data, can you provide me three specific reasons to explain the movie's performance? And why it received its rating?\n<|end|>\n"
    "<|assistant|>\n"
)

parser = StrOutputParser()

review_chain = review_prompt | llm | parser

analysis_chain = analysis_prompt | llm | parser

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
    sentiment = review_chain.invoke({"reviews": reviews_str})
    response = analysis_chain.invoke({
        "reviews": reviews_str,
        "system_prompt": system_prompt,
        "title": movie_data.title,
        "release_date": movie_data.release_date,
        "budget": movie_data.budget,
        "revenue": movie_data.revenue,
        "rating": movie_data.rating,
        "overview": movie_data.overview,
        "performance": performance,
        "sentiment": sentiment,
    })
    return {"analysis": response}
