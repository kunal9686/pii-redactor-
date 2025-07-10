import logging
import os
from datetime import datetime

# We're making a special diary called "logs" folder
LOGS_DIR = "logs"  # Our diary's name
os.makedirs(LOGS_DIR,exist_ok=True)  # Make sure the diary exists

# Each day gets a new page with today's date
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# How we write in our diary:
# - Date and time (asctime)
# - How important it is (levelname)
# - What happened (message)
logging.basicConfig(
    filename = LOG_FILE,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO,
)

# This gives everyone their own pen to write in the diary
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger