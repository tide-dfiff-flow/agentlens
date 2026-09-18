"""Unit tests for API module."""

import pytest
from fastapi.testclient import TestClient
from agentlens.src.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self):
        """Test health endpoint returns healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestExecuteEndpoint:
    """Tests for execute endpoint."""

    def test_execute_react_agent(self):
        """Test executing a ReAct agent."""
        response = client.post(
            "/execute",
            json={
                "agent_type": "react",
                "task": "What is 2 + 2?",
                "max_iterations": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "status" in data

    def test_execute_with_tools(self):
        """Test executing with tools."""
        response = client.post(
            "/execute",
            json={
                "agent_type": "react",
                "task": "Search for AI news",
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "running", "failed"]


class TestTraceEndpoints:
    """Tests for trace endpoints."""

    def test_list_traces(self):
        """Test listing traces."""
        response = client.get("/traces")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_traces_with_pagination(self):
        """Test listing traces with pagination."""
        response = client.get("/traces?limit=10&offset=0")
        assert response.status_code == 200


class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_list_metrics(self):
        """Test listing metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_stats(self):
        """Test getting aggregated stats."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_executions" in data


class TestDiagnoseEndpoint:
    """Tests for diagnose endpoint."""

    def test_diagnose_nonexistent_trace(self):
        """Test diagnosing a trace that doesn't exist."""
        response = client.post("/diagnose/nonexistent-id")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
