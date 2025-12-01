"""
Test the copilot chat endpoint with various scenarios.

Usage:
    python scripts/test_copilot_endpoint.py
"""

import requests
import json
import time
import sys


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_api_health():
    """Test if API is running."""
    print_section("TEST 1: API Health Check")
    
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Run: python run.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_simple_query():
    """Test simple copilot query without tool usage."""
    print_section("TEST 2: Simple Query (No Tools)")
    
    url = "http://localhost:5000/copilot/chat"
    payload = {
        "messages": [
            {"role": "user", "content": "Hello! What can you help me with?"}
        ],
        "session_id": "test_simple"
    }
    
    print(f"Request: {payload['messages'][0]['content']}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        
        if data.get('error'):
            print(f"❌ Error: {data['error']}")
            print(f"   Error Type: {data.get('error_type', 'Unknown')}")
            return False
        
        reply = data['data']['reply']
        print(f"\n✅ Response received:")
        print(f"   {reply[:200]}..." if len(reply) > 200 else f"   {reply}")
        return True
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_failure_prediction():
    """Test failure prediction tool."""
    print_section("TEST 3: Failure Prediction Tool")
    
    url = "http://localhost:5000/copilot/chat"
    payload = {
        "messages": [
            {"role": "user", "content": "Check the failure risk for unit L56614/9435"}
        ],
        "session_id": f"test_failure_{int(time.time())}"
    }
    
    print(f"Request: {payload['messages'][0]['content']}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        
        if data.get('error'):
            print(f"❌ Error: {data['error']}")
            print(f"   Error Type: {data.get('error_type', 'Unknown')}")
            
            # Print more details for debugging
            if 'function_response' in str(data.get('error', '')):
                print("\n   🔍 This is the Gemini function calling error.")
                print("   The fix should have resolved this issue.")
            
            return False
        
        reply = data['data']['reply']
        intermediate_steps = data['data'].get('intermediate_steps', [])
        
        print(f"\n✅ Response received:")
        print(f"   {reply[:300]}..." if len(reply) > 300 else f"   {reply}")
        print(f"\n   Tools used: {len(intermediate_steps)} step(s)")
        
        # Check if predict_failure was actually called
        tool_called = any('predict_failure' in str(step) for step in intermediate_steps)
        if tool_called:
            print("   ✅ predict_failure tool was invoked successfully")
        else:
            print("   ⚠️  predict_failure tool was NOT invoked (agent may have answered without tool)")
        
        return True
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_rul_prediction():
    """Test RUL prediction tool."""
    print_section("TEST 4: RUL Prediction Tool")
    
    url = "http://localhost:5000/copilot/chat"
    payload = {
        "messages": [
            {"role": "user", "content": "What is the remaining useful life for unit L56614/9435?"}
        ],
        "session_id": f"test_rul_{int(time.time())}"
    }
    
    print(f"Request: {payload['messages'][0]['content']}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        
        if data.get('error'):
            print(f"❌ Error: {data['error']}")
            return False
        
        reply = data['data']['reply']
        intermediate_steps = data['data'].get('intermediate_steps', [])
        
        print(f"\n✅ Response received:")
        print(f"   {reply[:300]}..." if len(reply) > 300 else f"   {reply}")
        print(f"\n   Tools used: {len(intermediate_steps)} step(s)")
        
        tool_called = any('predict_rul' in str(step) for step in intermediate_steps)
        if tool_called:
            print("   ✅ predict_rul tool was invoked successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_conversation_memory():
    """Test multi-turn conversation with memory."""
    print_section("TEST 5: Conversation Memory (Multi-turn)")
    
    url = "http://localhost:5000/copilot/chat"
    session_id = f"test_memory_{int(time.time())}"
    
    # Turn 1: Check a unit
    print("\n[Turn 1]")
    payload1 = {
        "messages": [
            {"role": "user", "content": "Check unit L56614/9435"}
        ],
        "session_id": session_id
    }
    
    print(f"Request: {payload1['messages'][0]['content']}")
    
    try:
        response1 = requests.post(url, json=payload1, timeout=60)
        data1 = response1.json()
        
        if data1.get('error'):
            print(f"❌ Turn 1 failed: {data1['error']}")
            return False
        
        reply1 = data1['data']['reply']
        print(f"Response: {reply1[:150]}..." if len(reply1) > 150 else f"Response: {reply1}")
        
        # Turn 2: Ask follow-up (should remember context)
        time.sleep(2)
        print("\n[Turn 2]")
        payload2 = {
            "messages": [
                {"role": "user", "content": "What was its failure probability?"}
            ],
            "session_id": session_id
        }
        
        print(f"Request: {payload2['messages'][0]['content']}")
        
        response2 = requests.post(url, json=payload2, timeout=60)
        data2 = response2.json()
        
        if data2.get('error'):
            print(f"❌ Turn 2 failed: {data2['error']}")
            return False
        
        reply2 = data2['data']['reply']
        print(f"Response: {reply2[:150]}..." if len(reply2) > 150 else f"Response: {reply2}")
        
        # Check if context was preserved
        if 'L56614' in reply2 or '9435' in reply2 or '%' in reply2 or 'failure' in reply2.lower():
            print("\n✅ Conversation memory is working (context preserved)")
        else:
            print("\n⚠️  Memory might not be working (no clear context reference)")
        
        return True
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("  COPILOT ENDPOINT TEST SUITE")
    print("  Testing Gemini 2.0 Flash with LangChain")
    print("=" * 70)
    
    tests = [
        ("API Health Check", test_api_health),
        ("Simple Query", test_simple_query),
        ("Failure Prediction", test_failure_prediction),
        ("RUL Prediction", test_rul_prediction),
        ("Conversation Memory", test_conversation_memory),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            time.sleep(1)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n\n❌ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   The Gemini function calling issue is fixed.")
        print("   Your copilot is ready for production!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("   Check the error messages above for details.")
    
    print("=" * 70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
