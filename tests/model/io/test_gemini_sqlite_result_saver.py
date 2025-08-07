import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, timezone

from model.io.gemini_sqlite_result_saver import GeminiSQLiteResultSaver


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        db_path = tmp.name
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


def test_database_schema(temp_db):
    """Test that the database is created with the correct schema."""
    # When
    saver = GeminiSQLiteResultSaver(temp_db)
    
    # Then
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(results)")
        columns = cursor.fetchall()
    
    # Verify column names and types
    expected_columns = [
        (0, 'id', 'INTEGER', 0, None, 1),
        (1, 'source_id', 'TEXT', 1, None, 0),
        (2, 'chunk_id', 'TEXT', 1, None, 0),
        (3, 'prompt', 'TEXT', 1, None, 0),
        (4, 'response', 'TEXT', 1, None, 0),
        (5, 'used_tokens', 'INTEGER', 0, None, 0),
        (6, 'model_version', 'TEXT', 1, None, 0),
        (7, 'timestamp', 'TEXT', 1, None, 0)
    ]
    
    assert len(columns) == len(expected_columns)
    for i, (cid, name, type_, notnull, dflt_value, pk) in enumerate(columns):
        assert (i, name, type_, notnull, dflt_value, pk) == expected_columns[i]


def test_save_and_retrieve_results(temp_db):
    """Test saving and retrieving results from the database."""
    # Given
    saver = GeminiSQLiteResultSaver(temp_db)
    test_results = [
        {
            'source_id': 'src1',
            'chunk_id': 'chk1',
            'prompt': 'Test prompt 1',
            'response': 'Test response 1',
            'model_version': 'gemini-1.0',
            'used_tokens': 100
        },
        {
            'source_id': 'src2',
            'chunk_id': 'chk2',
            'prompt': 'Test prompt 2',
            'response': 'Test response 2',
            'model_version': 'gemini-1.0',
            'used_tokens': 150
        }
    ]
    
    # When
    saver.save(test_results)
    
    # Then
    results = saver.get_all()
    
    assert len(results) == 2
    for i, result in enumerate(results):
        assert result['source_id'] == test_results[i]['source_id']
        assert result['chunk_id'] == test_results[i]['chunk_id']
        assert result['prompt'] == test_results[i]['prompt']
        assert result['response'] == test_results[i]['response']
        assert result['model_version'] == test_results[i]['model_version']
        assert result['used_tokens'] == test_results[i]['used_tokens']
        assert isinstance(result['id'], int)
        assert isinstance(result['timestamp'], str)
        # Verify timestamp is in ISO format with 'Z' timezone
        datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))


def test_save_empty_results_raises_error(temp_db):
    """Test that saving an empty list raises a ValueError."""
    # Given
    saver = GeminiSQLiteResultSaver(temp_db)
    
    # When/Then
    with pytest.raises(ValueError, match="No results to save"):
        saver.save([])


def test_save_with_optional_fields(temp_db):
    """Test saving results with optional fields."""
    # Given
    saver = GeminiSQLiteResultSaver(temp_db)
    test_result = {
        'source_id': 'src1',
        'chunk_id': 'chk1',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_version': 'gemini-1.0',
        # used_tokens is optional
    }
    
    # When
    saver.save([test_result])
    
    # Then
    results = saver.get_all()
    assert len(results) == 1
    assert results[0]['used_tokens'] is None
    assert results[0]['source_id'] == 'src1'


def test_required_fields_validation(temp_db):
    """Test that required fields are validated."""
    # Given
    saver = GeminiSQLiteResultSaver(temp_db)
    required_fields = ['source_id', 'chunk_id', 'prompt', 'response', 'model_version']
    
    for field in required_fields:
        # Create a test result missing one required field
        test_result = {
            'source_id': 'src1',
            'chunk_id': 'chk1',
            'prompt': 'Test prompt',
            'response': 'Test response',
            'model_version': 'gemini-1.0'
        }
        test_result.pop(field)
        
        # When/Then
        with pytest.raises(KeyError):
            saver.save([test_result])


def test_timestamp_is_utc(temp_db):
    """Test that timestamps are stored in UTC."""
    # Given
    saver = GeminiSQLiteResultSaver(temp_db)
    test_result = {
        'source_id': 'src1',
        'chunk_id': 'chk1',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_version': 'gemini-1.0'
    }
    
    # When
    before_save = datetime.now(timezone.utc)
    saver.save([test_result])
    after_save = datetime.now(timezone.utc)
    
    # Then
    results = saver.get_all()
    saved_timestamp = datetime.fromisoformat(results[0]['timestamp'].replace('Z', '+00:00'))
    
    assert before_save <= saved_timestamp <= after_save
    assert saved_timestamp.tzinfo == timezone.utc
