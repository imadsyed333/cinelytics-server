from pydantic import BaseModel

class MovieData(BaseModel):
    title: str
    release_year: int
    budget: float
    revenue: float
    overview: str
