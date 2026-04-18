from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import describe_performance, fetch_movie_data, system_prompt

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
    temperature=0.0,
    repetition_penalty=1.0,
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = PromptTemplate.from_template(
    "<|system|>\n{system_prompt}<|end|>\n"
    "<|user|>\nThe following is the data for a movie:\n\n{movie_json}\n\nThe movie's performance is categorized as: {performance}.\n\nBased on this data, provide specific reasons for why it performed the way it did. Use all available information and avoid making generic statements.<|end|>\n"
    "<|assistant|>\n"
)

parser = StrOutputParser()

chain = prompt_template | llm | parser

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

    performance = describe_performance(movie_data.revenue, movie_data.budget)
    response = chain.invoke({
        "system_prompt": system_prompt,
        "movie_json": movie_data.model_dump_json(),
        "performance": performance,
    })
    return {"analysis": response}
