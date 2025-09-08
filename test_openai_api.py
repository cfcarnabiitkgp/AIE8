#!/usr/bin/env python3
"""
Quick test script to verify OpenAI API key is working.
Run this script to check if your API key can successfully make requests to OpenAI.
"""

import os
import sys

from openai import OpenAI


def test_openai_api():
    """Test if OpenAI API key is working by making a simple request."""

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("\nTo set your API key, run one of these commands:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  # or add it to your ~/.zshrc file for persistence")
        return False

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        print("🔄 Testing OpenAI API connection...")

        # Make a simple test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Say 'API test successful' if you can read this.",
                }
            ],
            max_tokens=10,
        )

        # Check if we got a response
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message.content.strip()
            print(f"✅ API Key is working! Response: {message}")
            print(f"📊 Usage: {response.usage}")
            return True
        else:
            print("❌ No response received from API")
            return False

    except Exception as e:
        print(f"❌ API request failed: {str(e)}")

        # Provide helpful error messages
        if "Invalid API key" in str(e):
            print("\n💡 Your API key appears to be invalid. Please check:")
            print("  1. The key is correctly copied from OpenAI dashboard")
            print("  2. The key is properly set in your environment")
            print("  3. The key hasn't expired or been revoked")
        elif "Insufficient quota" in str(e):
            print("\n💡 You may have insufficient credits. Please check:")
            print("  1. Your OpenAI account has sufficient credits")
            print("  2. Your payment method is valid")
        elif "rate limit" in str(e).lower():
            print("\n💡 Rate limit exceeded. Please try again in a moment.")

        return False


if __name__ == "__main__":
    print("🚀 OpenAI API Key Test")
    print("=" * 30)

    success = test_openai_api()

    if success:
        print("\n🎉 Your OpenAI API is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ API test failed. Please check your setup.")
        sys.exit(1)
