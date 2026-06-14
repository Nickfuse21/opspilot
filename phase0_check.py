import os
from dotenv import load_dotenv
from google import genai  # new, supported SDK (replaces google.generativeai)

# 1. Load the .env file so GEMINI_API_KEY becomes available.
load_dotenv()

# 2. Read the key out of the environment (NOT hardcoded).
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("No GEMINI_API_KEY found - check your .env file")

# 3. Create one authenticated client (replaces configure + GenerativeModel).
client = genai.Client(api_key=api_key)

# 4. Send a tiny prompt; the model name is passed per-call now.
response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Reply with exactly: OpsPilot setup works!",
)
print("Gemini says:", response.text)
