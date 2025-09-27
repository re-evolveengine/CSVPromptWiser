# CSV PromptWiser

CSV PromptWiser is a Streamlit-based dashboard that lets you upload tabular data (CSV/Parquet), split it into manageable chunks, and process those chunks against an LLM (Google Gemini family) using a guided prompt. It emphasizes practical token budgeting, chunk management, and persistence of processed results.


## Key Features
- Upload or reuse a previously saved dataset.
- Configure chunking with recommended sizes based on your prompt and example response.
- Manage and visualize token budgets to avoid overruns.
- Process chunks with Gemini models and track live progress.
- Save processed results to a local SQLite DB and export them when ready.
- Polished Streamlit UX with collapsible panels and status summaries.


## Quickstart

1. Clone the repo
```bash
git clone https://github.com/<your-org-or-user>/CSVPromptWiser.git
cd CSVPromptWiser
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux (if applicable)
# source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set your API key
- The app expects an API key named `KEY`. You can either:
  - Enter it in the UI and choose "Save this key" (it will write to your `.env`), or
  - Create a `.env` file in the project root with:

```env
KEY=your_api_key_here
```

5. Run the app
```bash
streamlit run app.py
```


## How It Works (High Level)
- Entry point: `app.py` sets up the page and renders the main dashboard using Streamlit.
- Sidebar: `streamlit_dir/side_bar.py` wires together the controls:
  - LLM selection and model configuration
  - Data upload or reuse a previously saved file
  - Prompt and example response input
  - Chunking settings and token budgeting
  - Start processing and optional export of results
- Elements: `streamlit_dir/elements/` contains modular UI components such as:
  - `dataset_handler_ui.py` — load/upload datasets, chunking config, and chunk file creation
  - `chunk_processor_panel.py` — processing loop, live progress, token gauge, DB saving
  - `api_key_ui.py` — reads/saves `KEY` via `.env`
  - `model_selector_ui.py`, `llm_selector.py`, `prompt_input_ui.py`, etc.
- Constants: `utils/constants.py` defines `APP_NAME` (CSV PromptWiser), default settings, and storage locations.


## Storage and File Locations
To keep your working directory clean, the app stores runtime files under your user Documents folder:

- Documents directory used: `~/Documents/CSV PromptWiser/`
- `data/` — uploaded datasets
- `results/` — exported/processed results (SQLite DB and exports)
- `temp/` — transient files such as `chunks.json`
- `config/` — preferences (e.g., model prefs, prompt prefs)

These paths are defined in `utils/constants.py` (e.g., `DATA_DIR`, `RESULTS_DIR`, `TEMP_DIR`, `RESULTS_DB_PATH`).


## Typical Workflow
1. Choose an LLM and enter your API key in the sidebar.
2. Upload a CSV or Parquet file (or reuse the saved one).
3. Enter your prompt and an example response.
4. Configure chunking:
   - See recommended chunk size if a model is selected.
   - Set a token budget and chunk size.
   - Click "Chunk & Save" to generate `chunks.json` under `temp/`.
   - If prior results exist, the app will ask for confirmation before clearing the DB.
5. Configure processing:
   - Set number of chunks to process per run.
   - Click "Set Processing Parameters" to enable processing in the main panel.
6. In the main panel, click "Start Chunk Processing" to execute.
7. When results exist, use the export section to save processed outputs.


## Commands Reference
- Run the Streamlit app:
```bash
streamlit run app.py
```

- Run tests:
```bash
pytest -q
```
Note: The default `pytest.ini` includes coverage flags. If coverage paths do not match your local layout, you can run without coverage by using `-q` as shown above.


## Project Structure
```
CSVPromptWiser/
├─ app.py
├─ requirements.txt
├─ pytest.ini
├─ streamlit_dir/
│  ├─ side_bar.py
│  └─ elements/
│     ├─ api_key_ui.py
│     ├─ chunk_processor_panel.py
│     ├─ dataset_handler_ui.py
│     ├─ llm_selector.py
│     ├─ model_selector_ui.py
│     ├─ prompt_input_ui.py
│     ├─ render_chunking_warning_dialog.py
│     ├─ render_export_section.py
│     └─ token_usage_gauge.py
├─ model/
│  ├─ core/
│  │  ├─ chunk/
│  │  └─ llms/
│  └─ io/
├─ utils/
│  ├─ constants.py
│  ├─ providers.py
│  └─ ...
└─ tests/
   ├─ integration/
   └─ utils/
```


## Configuration Details
- App title and style: controlled by `utils/constants.py` (`APP_NAME`, `STREAMLIT_CSS_STYLES`).
- Token budgeting: handled via `utils/providers.get_model_prefs()` and shown in the sidebar and status panels.
- Chunking:
  - Creation and summary of `chunks.json` powered by `model.core.chunk` helpers through `dataset_handler_ui.py`.
  - Inspector: `model.core.chunk.chunk_json_inspector.ChunkJSONInspector`.
- Processing:
  - Loop managed by `streamlit_dir/elements/chunk_processor_panel.py`.
  - Each successful result is saved via `model.io.sqlite_result_saver.SQLiteResultSaver` and `model.io.save_processed_chunks_to_db.save_processed_chunk_to_db`.


## Troubleshooting
- No model options or failing API calls:
  - Ensure `KEY` is set in `.env` or entered in the sidebar and saved.
- Chunking button asks to clear DB:
  - This is expected if previous results exist; confirm to proceed.
- Token budget exceeded:
  - Reduce chunk size or increase your token budget in the sidebar.
- UI does not update after actions:
  - Streamlit reruns the script on widget interaction. If state seems stale, try the action again; the app manages state with `st.session_state`.


## Contributing
1. Fork the repository.
2. Create a feature branch.
3. Make your changes with tests.
4. Open a pull request describing your change and rationale.

Please keep functions small, name things descriptively, and avoid introducing unused imports.


## License
Add your preferred OSS license here, for example MIT.
