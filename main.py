from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

class MovieRequest(BaseModel):
    title: str
    budget: float
    box_office: float
    release_year: int

def build_prompt(data, performance):
    return f"""<|user|>
You are a film industry analyst.

Movie: {data.title}
Budget: ${data.budget}
Box Office: ${data.box_office}
Performance: {performance}
Release Year: {data.release_year}

Provide three reasons to explain why the movie performed the way it did. Be concise and realistic, drawing on industry trends and factors that typically influence a movie's success or failure.
<|end|>
<|assistant|>
"""

@app.post("/analyze")
def analyze(data: MovieRequest):
    ratio = data.box_office / data.budget
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
            max_new_tokens=300,
            temperature=0.4,
        )

    raw_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"analysis": raw_text}
