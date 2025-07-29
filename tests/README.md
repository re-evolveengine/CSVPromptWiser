# Test Suite

This directory contains the test suite for the PromptPilot application.

## Test Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions
- `conftest.py` - Pytest configuration and fixtures

## Running Tests

### Prerequisites

1. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

### Running All Tests

```bash
pytest
```

### Running Specific Tests

Run a single test file:
```bash
pytest tests/test_module.py
```

Run a specific test function:
```bash
pytest tests/test_module.py::test_function_name
```

### Test Coverage

To run tests with coverage report:
```bash
pytest --cov=src
```

### Running Tests in Parallel

```bash
pytest -n auto
```

### Test Report Generation

Generate HTML report:
```bash
pytest --html=report.html
```

## Writing Tests

- Test files should be named `test_*.py`
- Test functions should start with `test_`
- Test classes should start with `Test`
- Use fixtures for common test setup/teardown
- Keep tests isolated and independent
- Use descriptive test names
- Test both success and failure cases

## Best Practices

- Follow the Arrange-Act-Assert (AAA) pattern
- Use mocks for external dependencies
- Keep tests fast and focused
- Avoid test interdependencies
- Use parameterized tests for similar test cases
- Document complex test scenarios
