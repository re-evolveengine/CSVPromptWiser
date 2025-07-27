import os
import pandas as pd
import pytest
import dotenv


from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner


@pytest.mark.integration
def test_gemini_resilient_runner_with_dataframe():
    prompt = "Summarize the sales performance in this table:"

    df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Units Sold": [100, 150, 80],
        "Revenue": [1000, 2300, 900]
    })

    dotenv.load_dotenv()
    api_key = os.getenv("KEY")
    if not api_key:
        pytest.fail("Missing GEMINI_API_KEY in environment")

    client = GeminiClient(api_key=api_key, model="models/gemini-1.5-flash")
    runner = GeminiResilientRunner(client)

    response = runner.run(prompt, df)

    assert isinstance(response, str)
    assert any(product in response for product in ["Product A", "Product B", "Product C"])
    assert "units" in response.lower() or "revenue" in response.lower()
