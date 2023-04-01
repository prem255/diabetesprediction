import openai

# Set up the OpenAI API client
openai.api_key = "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"
model_engine = "text-curie-001"
prompt = "what is your name"

# Generate a response
response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the response
print(response.choices[0].text)
