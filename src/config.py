from pathlib import Path
from dotenv import dotenv_values
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / 'raw_data'
DATA_DIR = ROOT_DIR / 'data'
LOGS_DIR = ROOT_DIR / 'logs'

dotenv_path = '.env'
config = dotenv_values(dotenv_path)

MDC_API_KEY = config['MDC_API_KEY']
os.environ["MDC_DOWNLOAD_PATH"] = str(RAW_DATA_DIR)