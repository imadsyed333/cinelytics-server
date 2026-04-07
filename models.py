from pydantic import BaseModel

class MovieData(BaseModel):
    title: str
    release_year: int
    genre: str
    budget: float
    revenue: float
    overview: str
