from pydantic import BaseModel, Field
from typing import List, Optional

class BatchPredictionResponse(BaseModel):
    revenue_predictions: list[float]


class SingleMovieInput(BaseModel):
    id: Optional[float] = Field(None, description="Movie ID")
    belongs_to_collection: Optional[str] = Field(
        None, description="Stringified list/dict of collection data"
    )
    budget: Optional[int] = Field(None, description="Movie budget")
    genres: Optional[str] = Field(
        None, description="Stringified list of genres"
    )
    homepage: Optional[str] = Field(None, description="Homepage URL")
    imdb_id: Optional[str] = Field(None, description="IMDB ID")
    original_language: Optional[str] = Field(None, description="Original language code")
    original_title: Optional[str] = Field(None, description="Original title")
    overview: Optional[str] = Field(None, description="Overview of the movie")
    popularity: Optional[float] = Field(None, description="Popularity score")
    poster_path: Optional[str] = Field(None, description="Poster image path")
    production_companies: Optional[str] = Field(
        None, description="Stringified list of production companies"
    )
    production_countries: Optional[str] = Field(
        None, description="Stringified list of production countries"
    )
    release_date: Optional[str] = Field(None, description="Release date")
    runtime: Optional[float] = Field(None, description="Runtime in minutes")
    spoken_languages: Optional[str] = Field(
        None, description="Stringified list of spoken languages"
    )
    status: Optional[str] = Field(None, description="Release status")
    tagline: Optional[str] = Field(None, description="Tagline")
    title: str = Field(..., description="Movie title")
    Keywords: Optional[str] = Field(
        None, description="Stringified list of keywords"
    )
    cast: Optional[str] = Field(
        None, description="Stringified list of cast members"
    )
    crew: Optional[str] = Field(
        None, description="Stringified list of crew members"
    )
    revenue: Optional[int] = Field(None, description="Revenue (if present)")
    popularity2: Optional[float] = Field(None, description="Second popularity metric")
    rating: Optional[float] = Field(None, description="Movie rating")
    totalVotes: Optional[float] = Field(None, description="Total votes for rating")






class SinglePredictionResponse(BaseModel):
    revenue_prediction: float
