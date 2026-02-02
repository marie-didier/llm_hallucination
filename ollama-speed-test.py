import ollama
import time

print("Testing generation speed...")
start_time = time.time()

response = ollama.chat(model='llama3.2', messages=[
  {'role': 'user', 'content': 'Write a short poem about coding.'}
])

end_time = time.time()
duration = end_time - start_time

print(f"\n--- Stats ---")
print(f"Total Time: {duration:.2f} seconds")
print(f"Response Length: {len(response['message']['content'])} characters")