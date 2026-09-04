"""
Unit and integration tests for the FastAPI application and endpoints.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.app import app
from src.orchestration.agent_executor import AgentResponse


def run_all_tests():
    """Runs all API integration tests within a single application lifecycle."""
    print("Running API test suite...")

    with TestClient(app) as client:
        # 1. Test Documentation & OpenAPI
        docs_res = client.get("/docs")
        assert docs_res.status_code == 200, f"Expected 200 on /docs, got {docs_res.status_code}"

        openapi_res = client.get("/openapi.json")
        assert openapi_res.status_code == 200
        schema = openapi_res.json()
        assert "paths" in schema
        assert "/api/v1/query" in schema["paths"]
        assert "/api/v1/stream" in schema["paths"]
        assert "/health" in schema["paths"]

        # 2. Test Static Web Dashboard
        root_res = client.get("/")
        assert root_res.status_code == 200
        assert "ResearchAgent" in root_res.text

        css_res = client.get("/static/style.css")
        assert css_res.status_code == 200

        js_res = client.get("/static/app.js")
        assert js_res.status_code == 200
        print("✓ Static files and OpenAPI documentation verified.")

        # 3. Test Health, Metrics, and History
        health_res = client.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True
        assert health_data["indices_loaded"] is True
        assert "uptime_seconds" in health_data

        metrics_res = client.get("/api/v1/metrics")
        assert metrics_res.status_code == 200
        metrics_data = metrics_res.json()
        assert "total_queries_served" in metrics_data

        history_res = client.get("/api/v1/history")
        assert history_res.status_code == 200
        assert isinstance(history_res.json(), list)
        print("✓ Health, metrics, and session history endpoints verified.")

        # 4. Test Input Validation
        short_res = client.post("/api/v1/query", json={"query": "ab"})
        assert short_res.status_code == 422

        empty_res = client.post("/api/v1/query", json={})
        assert empty_res.status_code == 422
        print("✓ Pydantic request validation verified.")

        # 5. Test Synchronous Query Routing
        mock_executor = MagicMock()
        mock_executor.query.return_value = AgentResponse(
            answer="Attention is a sequence mixing mechanism [Evidence 1].",
            sources=["arXiv:1706.03762"],
            citations=["Evidence 1"],
            citation_ids=[1],
            confidence=1.0,
            is_grounded=True,
            missing_aspects=[],
            hallucination_flags=[],
            reasoning_trace=[{"step": "reader", "passages": 1}],
            iterations=1,
            query="What is attention in transformers?",
        )
        app.state.executor = mock_executor

        query_res = client.post(
            "/api/v1/query",
            json={"query": "What is attention in transformers?", "top_k": 5}
        )
        assert query_res.status_code == 200
        query_data = query_res.json()
        assert "Attention is a sequence mixing" in query_data["answer"]
        assert query_data["confidence"] == 1.0
        assert query_data["is_grounded"] is True
        assert query_data["citation_ids"] == [1]
        assert query_data["sources"] == ["arXiv:1706.03762"]
        print("✓ Query routing and response serialization verified.")

        # 6. Test Stream Generator Endpoint
        mock_executor.stream_query.return_value = [
            {"step": "retrieval_start", "message": "Searching...", "data": {}},
            {"step": "complete", "message": "Done", "data": query_data},
        ]
        stream_res = client.post(
            "/api/v1/stream",
            json={"query": "What is attention in transformers?", "top_k": 5}
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        print("✓ SSE Streaming endpoint verified.")

    print("\n🎉 ALL 6 API TEST SUITES PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
