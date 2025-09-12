import logging
import os
from datetime import datetime

# Define the log file name with a timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory path
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define the full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure the logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# --- THE FIX IS HERE ---
# Create a logger instance that other modules can import
logger = logging.getLogger("EngineSentinelLogger")

