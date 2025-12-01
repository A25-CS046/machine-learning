"""
Test Gemini API key validity.

Usage:
    python scripts/test_gemini_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed")

from langchain_google_genai import ChatGoogleGenerativeAI


def test_gemini_api():
    """Test if Gemini API key is valid."""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in environment")
        print("Please add it to your .env file:")
        print("  GEMINI_API_KEY=your_api_key_here")
        return False
    
    print(f"[INFO] Testing API key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=api_key,
            temperature=0.0
        )
        
        response = llm.invoke("Say 'API key is valid' if you can read this.")
        print(f"[SUCCESS] API key is valid!")
        print(f"[INFO] Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"[ERROR] API key test failed: {str(e)}")
        print("\nPossible issues:")
        print("  1. API key has expired - get a new one from https://aistudio.google.com/app/apikey")
        print("  2. API key is invalid - check for typos")
        print("  3. Gemini API is unavailable - try again later")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("GEMINI API KEY TEST")
    print("=" * 70)
    
    success = test_gemini_api()
    
    print("\n" + "=" * 70)
    if success:
        print("TEST PASSED")
        print("=" * 70)
        print("\nYour Gemini API is ready to use.")
        print("You can now run the copilot endpoint.")
    else:
        print("TEST FAILED")
        print("=" * 70)
        print("\nPlease fix the API key issue before using the copilot.")
    
    sys.exit(0 if success else 1)