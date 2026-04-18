from pydantic import BaseModel

class MovieData(BaseModel):
    title: str
    release_date: str
    budget: float
    revenue: float
    overview: str
