import streamlit as st
import logging

from model.io.csv_exporter import CSVExporter

logger = logging.getLogger(__name__)


# --- MODIFICATION 1: Add 'chunk_file_path' as a parameter ---
def render_export_section(chunk_file_path: str):
    """
    Renders the UI section for exporting processed results to a CSV file
    and provides a download button.
    """
    st.markdown("---")
    st.markdown("### 💾 Export Processed Results")
    st.info("This will merge all successfully processed records from the database with their original data.")

    # Input for the CSV filename
    file_name = st.text_input(
        "Enter CSV filename:",
        value="processed_results.csv",
        help="The name of the CSV file to be created."
    )

    # Button to trigger the export
    if st.button("Export to CSV"):
        if not file_name or not file_name.endswith(".csv"):
            st.error("Filename cannot be empty and must end with .csv")
            return  # Stop execution if filename is invalid

        try:
            # Use a spinner to show that work is being done
            with st.spinner(f"Exporting data to {file_name}..."):
                # --- MODIFICATION 2: Pass the path to the exporter ---
                exporter = CSVExporter(json_path=chunk_file_path)
                exporter.export_processed_with_original_rows(file_name)

            # Provide a download button upon success
            with open(file_name, "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=file_name,
                    mime="text/csv",
                )
            st.success(f"✅ Successfully exported data to `{file_name}`.")

        except (FileNotFoundError, ValueError) as e:
            st.error(f"❌ Export failed: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during export: {e}")
            logger.error(f"Failed to export CSV: {e}")