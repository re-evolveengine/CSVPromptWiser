import os
import tempfile
import pytest
from datetime import datetime
from model.io.gemini_sqlite_result_saver import GeminiSQLiteResultSaver


@pytest.fixture
def temp_db():
    # Create a temporary file for the database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        db_path = tmp.name

    # Create the saver with the temporary database
    saver = GeminiSQLiteResultSaver(db_path=db_path)

    yield saver, db_path

    # Clean up - close connections and remove the temporary file
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_init_creates_table(temp_db):
    saver, db_path = temp_db
    # Verify the table was created by checking if we can query it
    results = saver.get_all()
    assert isinstance(results, list)


def test_save_single_result(temp_db):
    saver, _ = temp_db

    test_data = [
        {
            'source_id': 'test1',
            'prompt': 'What is 2+2?',
            'response': '4'
        }
    ]

    saver.save(test_data)
    results = saver.get_all()

    assert len(results) == 1
    assert results[0]['source_id'] == 'test1'
    assert results[0]['prompt'] == 'What is 2+2?'
    assert results[0]['response'] == '4'
    assert 'timestamp' in results[0]


def test_save_multiple_results(temp_db):
    saver, _ = temp_db

    test_data = [
        {'source_id': f'test_{i}', 'prompt': f'prompt_{i}', 'response': f'response_{i}'}
        for i in range(3)
    ]

    saver.save(test_data)
    results = saver.get_all()

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result['source_id'] == f'test_{i}'
        assert result['prompt'] == f'prompt_{i}'
        assert result['response'] == f'response_{i}'


def test_save_empty_results_raises_error(temp_db):
    saver, _ = temp_db

    with pytest.raises(ValueError, match="No results to save"):
        saver.save([])


def test_get_all_returns_empty_list_when_no_results(temp_db):
    saver, _ = temp_db
    results = saver.get_all()
    assert results == []


def test_save_with_missing_required_fields(temp_db):
    saver, _ = temp_db

    with pytest.raises(KeyError):
        saver.save([{'source_id': 'test'}])  # Missing prompt and response

    with pytest.raises(KeyError):
        saver.save([{'prompt': 'test', 'response': 'test'}])  # Missing source_id
