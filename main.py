from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi.middleware.cors import CORSMiddleware

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
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16
)

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieRequest(BaseModel):
    title: str
    budget: float
    revenue: float
    release_date: str
    overview: str

def build_prompt(data, performance):
    return f"""<|system|>
You are Kowalski, the smartest of the Penguins of Madagascar, and an expert film industry analyst. You have been given the following information about a movie.<|end|>
<|user|>
Movie: {data.title}
Budget: {data.budget}
Revenue: {data.revenue}
Performance: {performance}
Release Date: {data.release_date}
Overview: {data.overview}

Based on this data, analyze the movie's performance and provide three specific reasons for why it performed the way it did. Use the movie's budget, title, revenue, release date, and overview to support your analysis. Avoid making generic statements and focus on specific factors that influenced the movie's success or failure. Be concise and insightful in your analysis. Do not include any information that is not directly supported by the provided data. Do not include introductory or concluding statements.
<|end|>
<|assistant|>
"""

@app.post("/analyze")
def analyze(data: MovieRequest):
    ratio = data.revenue / data.budget
    performance = (
        "Hit" if ratio >= 2.5 else
        "Moderate Success" if ratio >= 1.5 else
        "Break-even" if ratio >= 1.0 else
        "Underperformed"
    )

    prompt = build_prompt(data, performance)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
        )
    
    generated_tokens = output[0][inputs.input_ids.shape[-1]:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {"analysis": response}
