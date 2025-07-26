from cli_flow_controller import CLIFlowController
from model.core.chunker import DataFrameChunker
from model.utils.chunk_json_inspector import ChunkJSONInspector
from model.utils.constants import TEMP_DIR
from model.utils.data_prompt_arger import DataPromptArger
from model.utils.dataset_loader import DatasetLoader


    # loader = DatasetLoader()
    # df = loader.load('profiles.csv')
    # chunker = DataFrameChunker()
    # chunks = chunker.chunk_dataframe(df, chunk_size=4)
    # chunker.save_chunks_to_json(chunks)


if __name__ == "__main__":
    controller = CLIFlowController()
    controller.run()
