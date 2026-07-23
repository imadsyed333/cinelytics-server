from pydantic import BaseModel, Field

class Genre(BaseModel):
    name: str = Field(description="The genre category of the movie, e.g., 'Action', 'Drama', 'Comedy'. Used to identify the movie's thematic classification.")

class ProductionCompany(BaseModel):
    name: str = Field(description="The name of the production company that produced or co-produced the movie.")
    origin_country: str = Field(description="The country code where the production company is based, e.g., 'US', 'GB', 'FR'.")

class MovieData(BaseModel):
    title: str = Field(description="The official title of the movie.")
    release_date: str = Field(description="The theatrical release date of the movie in YYYY-MM-DD format.")
    budget: float = Field(description="The production budget of the movie in US dollars. Critical for calculating ROI and profitability.")
    revenue: float = Field(description="The total worldwide box office revenue in US dollars. Used to assess commercial success.")
    rating: float = Field(description="The average user rating (typically 0-10 scale). Indicates critical and audience reception quality.")
    overview: str = Field(description="A brief synopsis or description of the movie's plot and themes. Useful for understanding the movie's appeal.")

class MovieReview(BaseModel):
    author: str = Field(description="The name or username of the review author.")
    content: str = Field(description="The full text of the review. Contains detailed opinions, criticisms, and insights about the movie's quality and reception.")

class AnalysisResponse(BaseModel):
    performance_summary: str = Field(description="A concise summary of the movie's box office performance, e.g., 'The movie was a hit, generating 4x its budget in revenue.'")
    reasons: list[str] = Field(description="A list of specific reasons explaining the movie's box office performance.")
    final_thoughts: str = Field(description="A brief concluding statement summarizing the overall analysis.")