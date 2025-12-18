from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Box Office Analysis")

llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_batch=512,
    temperature=0.4,
    repeat_penalty=1.1,
    verbose=False,
)

class MovieRequest(BaseModel):
    title: str
    budget: float
    box_office: float
    franchise: bool

def classify_performance(budget, box_office):
    ratio = box_office / budget
    if ratio >= 2.5:
        return "Hit"
    if ratio >= 1.5:
        return "Moderate Success"
    if ratio >= 1.0:
        return "Break-even"
    return "Underperformed"

def build_prompt(data: MovieRequest, performance: str) -> str:
    return f"""[INST]
You are a film industry analyst.

Movie: {data.title}
Production Budget: ${data.budget} million
Worldwide Box Office: ${data.box_office} million
Performance: {performance}
Franchise Film: {"Yes" if data.franchise else "No"}

Provide 3 reasons explaining why the movie performed the way it did.

Be concise and realistic. Do not invent specific events or numbers.
[/INST]
"""

@app.post("/analyze")
def analyze_movie(data: MovieRequest):
    performance = classify_performance(data.budget, data.box_office)
    prompt = build_prompt(data, performance)

    output = llm(
        prompt,
        max_tokens=300,
        stop=["</s>"]
    )

    return {
        "movie": data.title,
        "performance": performance,
        "analysis": output["choices"][0]["text"].strip()
    }
