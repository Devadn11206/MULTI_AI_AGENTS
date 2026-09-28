import os
import sys

# Ensure the root directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm.manager import get_llm_provider

def main():
    print("--- MANUAL GEMINI API TEST ---")
    print("WARNING: This makes a real API request to Gemini if GEMINI_API_KEY is set in .env.")
    
    try:
        provider = get_llm_provider()
        print(f"Initialized provider: {provider.__class__.__name__}")
        
        prompt = "Hello, what is 2 + 2? Please answer in one sentence."
        print(f"\nSending prompt: '{prompt}'")
        
        response = provider.generate(prompt=prompt)
        print(f"\nResponse:\n{response}")
        
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
