"""
Test script for Azure OpenAI API via APIM Gateway
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Configuration
APIM_ENDPOINT = os.getenv("APIM_ENDPOINT", "https://apim-dev-genaishared-3dr5cskzi3l5q.azure-api.net")
API_VERSION = "2024-02-15-preview"
DEPLOYMENT_NAME = "gpt-4o-mini-2024-07-18"  # Change to your deployment name

def test_chat_completion():
    """Test chat completion using Azure OpenAI API through APIM with managed identity"""
    
    # Use DefaultAzureCredential for authentication
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )
    
    # Initialize Azure OpenAI client with APIM endpoint
    client = AzureOpenAI(
        azure_endpoint=APIM_ENDPOINT,
        api_version=API_VERSION,
        azure_ad_token_provider=token_provider
    )
    
    try:
        # Make a chat completion request
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You always answer in detail."},
                {"role": "user", "content": "What is Azure API Management? Please give me a detailed breakdown of the main features."}
            ],
            temperature=0.7
        )
        
        # Print the response
        print("✓ Chat completion successful!")
        print(f"\nModel: {response.model}")
        print(f"\nResponse:\n{response.choices[0].message.content}")
        print(f"\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during chat completion: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Azure OpenAI API via APIM Gateway")
    print("=" * 60)
    print(f"Endpoint: {APIM_ENDPOINT}")
    print(f"Deployment: {DEPLOYMENT_NAME}")
    print("=" * 60)
    
    # Run tests in parallel
    test_results = []
    
    # Use ThreadPoolExecutor to run tests in parallel (max 10 concurrent requests)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {executor.submit(test_chat_completion): i for i in range(100)}
        
        # Collect results as they complete
        for idx, future in enumerate(as_completed(futures), 1):
            print(f"\n[Test {futures[future]+1}/100] Completed")
            test_results.append(future.result())
    
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {sum(test_results)}/{len(test_results)} tests passed")
    print("=" * 60)
    
    exit(0 if all(test_results) else 1)
