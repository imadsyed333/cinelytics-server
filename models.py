from pydantic import BaseModel

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