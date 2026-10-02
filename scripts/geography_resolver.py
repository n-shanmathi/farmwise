import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "output" / "geography_numpy_index.npz"
METADATA_PATH = PROJECT_ROOT / "output" / "geography_numpy_metadata.csv"
RESULTS_PATH = PROJECT_ROOT / "output" / "geography_resolver_test_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "output" / "geography_resolver_test_results.json"
ENV_PATH = PROJECT_ROOT / ".env"
REGION = "ap-south-1"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
MINISTRAL_MODEL_ID = "mistral.ministral-3-8b-instruct"
EMBEDDING_DIMENSION = 1024
MAX_TESTS = 5
TOP_K = 5
RESULT_COLUMNS = [
    "query", "extracted_location", "state", "district", "taluk", "block", "village",
    "resolved_place_id", "resolved_place_name", "resolved_place_type", "latitude",
    "longitude", "similarity_score", "confidence", "expected_place_id",
    "expected_place_type", "correct", "candidates",
]


def clean_value(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Ministral response did not contain JSON.")
        return json.loads(match.group(0))


def normalize_extracted(extracted):
    return {
        "location_mention": clean_value(extracted.get("location_mention")),
        "state": clean_value(extracted.get("state")),
        "district": clean_value(extracted.get("district")),
        "taluk": clean_value(extracted.get("taluk")),
        "block": clean_value(extracted.get("block")),
        "village": clean_value(extracted.get("village")),
        "place_type": clean_value(extracted.get("place_type")),
    }


def extraction_prompt(queries):
    numbered = "\n".join(f"{index}. {query}" for index, query in enumerate(queries, start=1))
    return (
        "Extract geography fields for each numbered user query. Return a JSON array in the same order, "
        "with exactly these keys per item: location_mention, state, district, taluk, block, village, "
        "place_type. Use null for fields not explicitly supported. Do not invent names.\n\n"
        f"Queries:\n{numbered}"
    )


def extract_locations(client, queries):
    response = client.invoke_model(
        modelId=MINISTRAL_MODEL_ID,
        body=json.dumps({
            "messages": [{"role": "user", "content": extraction_prompt(queries)}],
            "max_tokens": 320,
            "temperature": 0,
            "top_p": 1,
        }),
        contentType="application/json",
        accept="application/json",
    )
    response_data = json.loads(response["body"].read())
    generated = response_data["choices"][0]["message"]["content"]
    extracted = extract_json(generated)
    if not isinstance(extracted, list) or len(extracted) != len(queries):
        raise ValueError("Ministral batch extraction did not return one item per query.")
    return [normalize_extracted(item) for item in extracted]


def extract_location(client, query):
    prompt = (
        "Extract geography fields from this user query. Return JSON only, with exactly these keys: "
        "location_mention, state, district, taluk, block, village, place_type. "
        "Use null for fields not explicitly supported by the query. Do not invent names.\n\n"
        f"User query: {query}"
    )
    response = client.invoke_model(
        modelId=MINISTRAL_MODEL_ID,
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 180,
            "temperature": 0,
            "top_p": 1,
        }),
        contentType="application/json",
        accept="application/json",
    )
    response_data = json.loads(response["body"].read())
    generated = response_data["choices"][0]["message"]["content"]
    extracted = extract_json(generated)
    return normalize_extracted(extracted)


def query_embedding(client, query):
    response = client.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": query}),
        contentType="application/json",
        accept="application/json",
    )
    vector = np.asarray(json.loads(response["body"].read())["embedding"], dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError("Titan query embedding does not have 1024 dimensions.")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Titan query embedding is a zero vector.")
    return vector / norm


def normalized(value):
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def name_match(extracted, candidate):
    names = [extracted.get("location_mention"), extracted.get("village"), extracted.get("block"), extracted.get("taluk"), extracted.get("district"), extracted.get("state")]
    candidate_names = [candidate.get("place_name"), candidate.get("village_name"), candidate.get("block_name"), candidate.get("taluk_name"), candidate.get("district_name"), candidate.get("state_name")]
    extracted_names = {normalized(name) for name in names if name}
    return int(bool(extracted_names & {normalized(name) for name in candidate_names if clean_value(name)}))


def hierarchy_match(extracted, candidate):
    comparisons = [("state", "state_name"), ("district", "district_name"), ("taluk", "taluk_name"), ("block", "block_name"), ("village", "village_name")]
    specified = [(left, right) for left, right in comparisons if extracted.get(left)]
    if not specified:
        return 0
    return int(all(normalized(extracted[left]) == normalized(candidate[right]) for left, right in specified if candidate.get(right)))


def resolve_candidate(extracted, candidate, similarity):
    semantic_score = (float(similarity) + 1.0) / 2.0
    exact_name = name_match(extracted, candidate)
    type_match = int(bool(extracted.get("place_type")) and normalized(extracted["place_type"]) == normalized(candidate["place_type"]))
    hierarchy = hierarchy_match(extracted, candidate)
    confidence = 0.60 * semantic_score + 0.25 * exact_name + 0.10 * type_match + 0.05 * hierarchy
    return round(float(confidence), 6)


class GeographyResolver:
    def __init__(self, runtime=None):
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        self.index = np.load(INDEX_PATH)["vectors"]
        self.metadata = pd.read_csv(METADATA_PATH, dtype=str)
        if self.index.shape != (len(self.metadata), EMBEDDING_DIMENSION):
            raise ValueError("Validated index and metadata shape mismatch.")
        self.runtime = runtime or boto3.client("bedrock-runtime", region_name=REGION)

    def resolve(self, query):
        extracted = extract_location(self.runtime, query)
        vector = query_embedding(self.runtime, query)
        similarities = self.index @ vector
        top_indexes = np.argsort(-similarities)[:TOP_K]
        candidates = []
        for vector_rank, index_id in enumerate(top_indexes, start=1):
            candidate = self.metadata.iloc[index_id].to_dict()
            candidate["vector_rank"] = vector_rank
            candidate["similarity_score"] = round(float(similarities[index_id]), 6)
            candidate["resolver_score"] = resolve_candidate(extracted, candidate, similarities[index_id])
            candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (candidate["resolver_score"], candidate["similarity_score"]),
            reverse=True,
        )
        ambiguous = (
            len(candidates) > 1
            and candidates[0]["resolver_score"] - candidates[1]["resolver_score"] < 0.04
        ) or candidates[0]["resolver_score"] < 0.65
        return extracted, candidates, None if ambiguous else candidates[0], ambiguous


def test_cases(metadata):
    cases = []
    queries = {
        "STATE": "Where is Tamil Nadu located?",
        "DISTRICT": "Which district is Chennai in Tamil Nadu?",
        "TALUK": "I need information about Sendurai taluk in Ariyalur district.",
        "BLOCK": "Show me Acharapakkam block in Chengalpattu district.",
        "VILLAGE": "Where is Gummidipoondi Tp in Gummidipoondi taluk, Thiruvallur district?",
    }
    for place_type in ("STATE", "DISTRICT", "TALUK", "BLOCK", "VILLAGE"):
        record = metadata[metadata["place_type"] == place_type].sort_values("place_id").iloc[0]
        cases.append({"query": queries[place_type], "expected_place_id": record["place_id"], "expected_place_type": place_type})
    return cases


def print_manual_result(query, extracted, vector_candidates, resolved, ambiguous):
    print("\n" + "=" * 72)
    print("Original query:", query)
    print("\nMinistral extracted:")
    print("  Location mention:", extracted.get("location_mention"))
    for field in ("state", "district", "taluk", "block", "village"):
        print(f"  {field.title()}:", extracted.get(field))
    print("  Place type:", extracted.get("place_type"))
    print("\nVector search Top-5:")
    for candidate in vector_candidates:
        print(
            f"  Rank {candidate['vector_rank']}: {candidate['place_id']} | "
            f"{candidate['place_name']} | {candidate['place_type']} | "
            f"similarity={candidate['similarity_score']:.6f}"
        )
    print("\nFinal resolver selection:")
    if ambiguous:
        print("  AMBIGUOUS: no canonical place selected")
        print("  Resolver confidence is a score, not a probability.")
        return
    print("  Place ID:", resolved["place_id"])
    print("  Place name:", resolved["place_name"])
    print("  Place type:", resolved["place_type"])
    print("  District:", resolved.get("district_name"))
    print("  Taluk:", resolved.get("taluk_name"))
    print("  Block:", resolved.get("block_name"))
    print("  Village:", resolved.get("village_name"))
    print("  Latitude:", resolved.get("latitude"))
    print("  Longitude:", resolved.get("longitude"))
    print("  Similarity score:", resolved["similarity_score"])
    print("  Resolver confidence score:", resolved["resolver_score"])
    if extracted.get("place_type") and normalized(extracted["place_type"]) != normalized(resolved["place_type"]):
        print("  Note: extracted place type differs from the canonical dataset type.")


def manual_main():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    index = np.load(INDEX_PATH)["vectors"]
    metadata = pd.read_csv(METADATA_PATH, dtype=str)
    if index.shape != (len(metadata), EMBEDDING_DIMENSION):
        raise ValueError("Validated index and metadata shape mismatch.")
    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    print("Manual Geography Resolver mode. Maximum queries: 10.")
    print("Type 'exit' or 'quit' to stop. Each accepted query requires one confirmation before AWS calls.")
    processed = 0
    while processed < 10:
        query = input("\nGeography query: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        processed += 1
        confirmation = input("Run Ministral extraction and Titan query embedding? [y/N]: ").strip().lower()
        if confirmation not in {"y", "yes"}:
            print("Skipped without AWS calls.")
            continue
        extracted = extract_location(runtime, query)
        vector = query_embedding(runtime, query)
        similarities = index @ vector
        top_indexes = np.argsort(-similarities)[:TOP_K]
        vector_candidates = []
        for vector_rank, index_id in enumerate(top_indexes, start=1):
            candidate = metadata.iloc[index_id].to_dict()
            candidate["vector_rank"] = vector_rank
            candidate["similarity_score"] = round(float(similarities[index_id]), 6)
            candidate["resolver_score"] = resolve_candidate(extracted, candidate, similarities[index_id])
            vector_candidates.append(candidate)
        reranked = sorted(
            vector_candidates,
            key=lambda candidate: (candidate["resolver_score"], candidate["similarity_score"]),
            reverse=True,
        )
        ambiguous = (
            len(reranked) > 1
            and reranked[0]["resolver_score"] - reranked[1]["resolver_score"] < 0.04
        ) or reranked[0]["resolver_score"] < 0.65
        print_manual_result(query, extracted, vector_candidates, None if ambiguous else reranked[0], ambiguous)
    print(f"Manual session ended after {processed} query slot(s).")


def main():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    index = np.load(INDEX_PATH)["vectors"]
    metadata = pd.read_csv(METADATA_PATH, dtype=str)
    if index.shape != (len(metadata), EMBEDDING_DIMENSION):
        raise ValueError("Validated index and metadata shape mismatch.")

    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    cases = test_cases(metadata)
    output_rows = []
    extraction_records = []
    extracted_map = dict(zip(
        [case["query"] for case in cases[:2]],
        extract_locations(runtime, [case["query"] for case in cases[:2]]),
    ))
    ministral_calls = 1
    for case in cases[2:MAX_TESTS]:
        extracted_map[case["query"]] = extract_location(runtime, case["query"])
        ministral_calls += 1
    for case in cases[:MAX_TESTS]:
        extracted = extracted_map[case["query"]]
        vector = query_embedding(runtime, case["query"])
        similarities = index @ vector
        top_indexes = np.argsort(-similarities)[:TOP_K]
        candidates = []
        for index_id in top_indexes:
            candidate = metadata.iloc[index_id].to_dict()
            candidate["similarity_score"] = round(float(similarities[index_id]), 6)
            candidate["resolver_score"] = resolve_candidate(extracted, candidate, similarities[index_id])
            candidates.append(candidate)
        candidates.sort(key=lambda candidate: (candidate["resolver_score"], candidate["similarity_score"]), reverse=True)
        resolved = candidates[0]
        extraction_records.append({"query": case["query"], "extracted": extracted})
        output_rows.append({
            "query": case["query"],
            "extracted_location": extracted["location_mention"],
            "state": extracted["state"], "district": extracted["district"], "taluk": extracted["taluk"],
            "block": extracted["block"], "village": extracted["village"],
            "resolved_place_id": resolved["place_id"], "resolved_place_name": resolved["place_name"],
            "resolved_place_type": resolved["place_type"], "latitude": resolved["latitude"],
            "longitude": resolved["longitude"], "similarity_score": resolved["similarity_score"],
            "confidence": resolved["resolver_score"], "expected_place_id": case["expected_place_id"],
            "expected_place_type": case["expected_place_type"], "correct": resolved["place_id"] == case["expected_place_id"],
            "candidates": json.dumps(candidates, separators=(",", ":")),
        })

    results = pd.DataFrame(output_rows, columns=RESULT_COLUMNS)
    results.to_csv(RESULTS_PATH, index=False)
    summary = {
        "model_id_ministral": MINISTRAL_MODEL_ID,
        "model_id_titan": TITAN_MODEL_ID,
        "region": REGION,
        "query_count": len(output_rows),
        "ministral_calls": ministral_calls,
        "titan_query_calls": len(output_rows),
        "vector_count": len(index),
        "top_1_correct": int(results["correct"].sum()),
        "top_1_incorrect": int((~results["correct"]).sum()),
        "confidence_formula": "0.60*((cosine_similarity+1)/2) + 0.25*name_match + 0.10*place_type_match + 0.05*hierarchy_match",
        "results": output_rows,
        "extractions": extraction_records,
        "limitations": ["Five-query prototype only; confidence is a transparent resolver score, not a calibrated probability."],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=4) + "\n", encoding="utf-8")
    print("Queries tested:", len(output_rows))
    print("Ministral calls:", len(output_rows))
    print("Titan query calls:", len(output_rows))
    for row in output_rows:
        print(f"Query: {row['query']} | Expected: {row['expected_place_id']} | Resolved: {row['resolved_place_id']} | Type: {row['resolved_place_type']} | Score: {row['similarity_score']} | Confidence: {row['confidence']} | Correct: {row['correct']}")
    print("Results created:", RESULTS_PATH)
    print("Summary created:", SUMMARY_PATH)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Geography Resolver validation and manual CLI.")
        parser.add_argument("--manual", action="store_true", help="Run interactive resolver mode.")
        arguments = parser.parse_args()
        manual_main() if arguments.manual else main()
    except (BotoCoreError, ClientError, KeyError, ValueError, json.JSONDecodeError) as error:
        print("Geography resolver validation failed:", error)