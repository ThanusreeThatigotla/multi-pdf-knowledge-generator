from google import genai

api_key = "YOUR_API_KEY"

client = genai.Client(api_key=api_key)

prompt = input("Enter prompt: ")

response = client.models.generate_content(
    model="gemini-1.5-flash-latest",
    contents=prompt
)

print(response.text)