import os
from cli.cli_flow_controller import CLIFlowController
from model.utils.constants import CONFIG_DIR_CLI, APP_DIR_CLI, DATA_DIR_CLI, RESULTS_DIR_CLI, TEMP_DIR_CLI

# Ensure all required directories exist
for directory in [APP_DIR_CLI, CONFIG_DIR_CLI, DATA_DIR_CLI, RESULTS_DIR_CLI, TEMP_DIR_CLI]:
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    controller = CLIFlowController()
    controller.run()
