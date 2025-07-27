from cli_flow_controller import CLIFlowController

# loader = DatasetLoader()
    # df = loader.load('profiles.csv')
    # chunker = DataFrameChunker()
    # chunks = chunker.chunk_dataframe(df, chunk_size=4)
    # chunker.save_chunks_to_json(chunks)


if __name__ == "__main__":
    controller = CLIFlowController()
    controller.run()
