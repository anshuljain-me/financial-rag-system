from app.core.config import get_settings
from google import genai

settings = get_settings()

# Strip any accidental whitespace or quotes from the key
api_key = settings.GEMINI_API_KEY.strip().strip("'").strip('"')
print(f"1. API Key Loaded: {api_key[:6]}...{api_key[-4:]} (Length: {len(api_key)})")

client = genai.Client(api_key=api_key)

print("\n2. Querying Google AI Studio for Available Models...")
try:
    models_list = list(client.models.list())
    print(f"   Found {len(models_list)} models.")
    for m in models_list:
        # Display the clean model ID
        clean_name = m.name.replace("models/", "")
        print(f"   • {clean_name}")
except Exception as e:
    print(f"   ❌ Could not list models: {e}")

print("\n3. Testing Content Generation...")
candidate_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

for model_id in candidate_models:
    try:
        print(f"   Attempting model: '{model_id}'...")
        response = client.models.generate_content(
            model=model_id,
            contents="Ping. Respond with 'Financial RAG Ready' only."
        )
        print(f"   ✅ SUCCESS with '{model_id}'! Response: {response.text.strip()}")
        break
    except Exception as e:
        print(f"   ❌ Failed with '{model_id}': {e}\n")