from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.geography_resolver import GeographyResolver


class ResolveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


def clean_json_value(value):
    if value is None:
        return None
    try:
        return None if value != value else value
    except TypeError:
        return value


def serialize_candidate(candidate):
    return {key: clean_json_value(value) for key, value in candidate.items()}


@asynccontextmanager
async def lifespan(app):
    app.state.resolver = GeographyResolver()
    yield


app = FastAPI(
    title="Geography Resolver API",
    description="REST API for the validated Phase 10G geography resolver.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request):
    resolver = request.app.state.resolver
    return {
        "status": "healthy",
        "service": "geography-resolver",
        "records": len(resolver.metadata),
    }


@app.post("/resolve")
def resolve(request: Request, payload: ResolveRequest):
    try:
        extracted, candidates, resolved, ambiguous = request.app.state.resolver.resolve(payload.query)
    except Exception as error:
        raise HTTPException(status_code=502, detail="Geography resolver backend failed.") from error

    response = {
        "query": payload.query,
        "extracted_location": extracted["location_mention"],
        "state": extracted["state"],
        "district": extracted["district"],
        "taluk": extracted["taluk"],
        "block": extracted["block"],
        "village": extracted["village"],
        "resolved_place_id": None,
        "resolved_place_name": None,
        "resolved_place_type": None,
        "latitude": None,
        "longitude": None,
        "similarity": None,
        "confidence": None,
        "ambiguous": ambiguous,
        "candidates": [serialize_candidate(candidate) for candidate in candidates],
    }
    if resolved is not None:
        response.update({
            "resolved_place_id": resolved["place_id"],
            "resolved_place_name": resolved["place_name"],
            "resolved_place_type": resolved["place_type"],
            "latitude": clean_json_value(resolved["latitude"]),
            "longitude": clean_json_value(resolved["longitude"]),
            "similarity": resolved["similarity_score"],
            "confidence": resolved["resolver_score"],
        })
    return response