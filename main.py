from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
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
    franchise: bool

def build_prompt(data, performance):
    return f"""[INST]
You are a film industry analyst.

Movie: {data.title}
Budget: ${data.budget}M
Box Office: ${data.box_office}M
Performance: {performance}
Franchise: {"Yes" if data.franchise else "No"}

Provide three reasons to explain why the movie performed the way it did.
Be concise and realistic.
[/INST]
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
            do_sample=True
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"analysis": text}
