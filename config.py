import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
RAW_CACHE_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Default Excel path
EXCEL_PATH = INPUT_DIR / "companies.xlsx"

# Channel names (also the expected sheet names in the Excel file)
CHANNELS = ["Builders Merchants", "Plumbing Merchants", "Department Stores"]

# Companies House API
API_BASE_URL = "https://api.company-information.service.gov.uk"
API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY", "")

# Rate limiting: 600 requests per 5 minutes
RATE_LIMIT_CALLS = 600
RATE_LIMIT_WINDOW = 300  # seconds

# Ensure directories exist
for d in [INPUT_DIR, RAW_CACHE_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
