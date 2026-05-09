from pydantic import BaseModel, Field

class MovieData(BaseModel):
    title: str
    release_date: str
    budget: float
    revenue: float
    rating: float
    overview: str

class MovieReview(BaseModel):
    id: str
    author: str
    content: str

class AnalysisResponse(BaseModel):
    performance_summary: str = Field(description="A concise summary of the movie's box office performance, e.g., 'The movie was a hit, generating 4x its budget in revenue.'")
    reasons: list[str] = Field(description="A list of three specific reasons explaining the movie's box office performance.")
    final_thoughts: str = Field(description="A brief concluding statement summarizing the overall analysis.")