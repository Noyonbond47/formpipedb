from fastapi.testclient import TestClient
from unittest.mock import MagicMock, Mock
import pytest
import os
import sys

# Add the project root to the Python path to allow imports from the 'api' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app, get_current_user_details
from postgrest import APIError

# The TestClient allows us to make requests to our FastAPI app without a running server
client = TestClient(app)

# --- Mocking Authentication ---

# 1. Create a mock user object that mimics what Supabase returns.
mock_user = Mock()
mock_user.id = "mock-user-id-12345"

# 2. Create a mock Supabase client. MagicMock allows us to simulate chained calls
#    like `supabase.table(...).select(...).execute()`.
mock_supabase_client = MagicMock()

# 3. This function will be used to override the real `get_current_user_details` dependency.
#    Instead of requiring a real token, it will just return our mock user and client.
async def override_get_current_user_details():
    return {"user": mock_user, "client": mock_supabase_client}

# 4. Apply the override to the FastAPI app. Now, any endpoint that depends on
#    `get_current_user_details` will receive our mock data instead.
app.dependency_overrides[get_current_user_details] = override_get_current_user_details


# --- API Tests ---

@pytest.fixture(autouse=True)
def reset_mocks():
    """
    A pytest fixture that automatically runs before each test.
    It resets the mock client to ensure tests are isolated from each other.
    """
    mock_supabase_client.reset_mock()

def test_get_user_databases_success():
    """
    Tests the successful retrieval of a user's databases.
    """
    # Arrange: Configure the mock client's return value for this specific test.
    mock_db_list = [
        {"id": 1, "created_at": "2023-01-01T00:00:00Z", "name": "test_db_1", "description": "First test db"},
        {"id": 2, "created_at": "2023-01-02T00:00:00Z", "name": "test_db_2", "description": "Second test db"},
    ]
    # This simulates the full chain: supabase.table(...).select(...).order(...).execute()
    # and sets the 'data' attribute on the final returned object.
    mock_supabase_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = mock_db_list

    # Act: Make a request to the endpoint.
    response = client.get("/api/v1/databases")
    
    # Assert: Check that the response is what we expect.
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data) == 2
    assert response_data[0]["name"] == "test_db_1"
    # Verify that the mock was called correctly
    mock_supabase_client.table.return_value.select.assert_called_with("id, created_at, name, description")

def test_get_single_database_success():
    """
    Tests the successful retrieval of a single, authorized database.
    """
    # Arrange
    mock_db = {"id": 1, "created_at": "2023-01-01T00:00:00Z", "name": "test_db_1", "description": "First test db"}
    mock_supabase_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = mock_db

    # Act
    response = client.get("/api/v1/databases/1")

    # Assert
    assert response.status_code == 200
    assert response.json()["name"] == "test_db_1"
    mock_supabase_client.table.return_value.select.return_value.eq.assert_called_with("id", 1)

def test_get_single_database_not_found_or_unauthorized():
    """
    Tests that fetching a non-existent or unauthorized database returns 404.
    This simulates the behavior of RLS, where a query for another user's data
    returns an empty result, triggering the API's 404 handling.
    """
    # Arrange: Configure the mock to return None, as if RLS filtered the result.
    mock_supabase_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None

    # Act
    response = client.get("/api/v1/databases/999") # 999 is an ID that doesn't exist or isn't owned by the user

    # Assert
    assert response.status_code == 404
    assert "Database not found" in response.json()["detail"]
    # Verify that the mock was called with the correct database ID
    mock_supabase_client.table.return_value.select.return_value.eq.assert_called_with("id", 999)

def test_create_user_database_success():
    """
    Tests the successful creation of a new database.
    """
    # Arrange
    mock_created_db = {
        "id": 3,
        "created_at": "2023-01-03T00:00:00Z",
        "name": "new_db",
        "description": "A new db"
    }
    mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [mock_created_db]

    db_payload = {"name": "new_db", "description": "A new db"}
    
    # Act
    response = client.post("/api/v1/databases", json=db_payload)

    # Assert
    assert response.status_code == 201
    response_data = response.json()
    assert response_data["name"] == "new_db"
    assert response_data["id"] == 3

def test_create_user_database_conflict():
    """
    Tests the error handling when creating a database with a name that already exists.
    """
    # Arrange: Configure the mock to raise a specific APIError.
    # We use `side_effect` to make the mock raise an exception when called.
    mock_api_error = APIError({
        "message": "duplicate key value violates unique constraint", "code": "23505"
    })
    mock_supabase_client.table.return_value.insert.return_value.execute.side_effect = mock_api_error

    db_payload = {"name": "existing_db", "description": "This one exists"}
    
    # Act
    response = client.post("/api/v1/databases", json=db_payload)

    # Assert
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
