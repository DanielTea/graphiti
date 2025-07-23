#!/usr/bin/env python3
"""
Graphiti Neo4j Connection Test Script

This script tests your Neo4j connection and environment setup
before running the main quickstart example.
"""

import asyncio
import os
import sys
from neo4j import AsyncGraphDatabase


async def test_neo4j_connection():
    """Test Neo4j database connection."""
    print("🔌 Testing Neo4j connection...")
    
    # Get connection parameters from environment
    uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD', 'password')
    
    print(f"   URI: {uri}")
    print(f"   User: {user}")
    print(f"   Password: {'*' * len(password) if password else 'Not set'}")
    print()
    
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        result = await driver.execute_query('RETURN "Connection successful!" as message, datetime() as timestamp')
        records = result.records
        
        if records:
            message = records[0]['message']
            timestamp = records[0]['timestamp']
            print(f"✅ {message}")
            print(f"   Server timestamp: {timestamp}")
        
        await driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed:")
        print(f"   Error: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print("   1. Make sure Neo4j is running (check Neo4j Desktop)")
        print("   2. Verify your password is correct")
        print("   3. Check if Neo4j is running on a different port")
        print("   4. Try accessing http://localhost:7474 in your browser")
        return False


def test_openai_key():
    """Test OpenAI API key configuration."""
    print("🔑 Testing OpenAI API key...")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY=your_api_key")
        return False
    
    if api_key == 'your_openai_api_key_here' or api_key == 'your_actual_openai_api_key':
        print("❌ OPENAI_API_KEY is set to placeholder value")
        print("   Replace with your actual OpenAI API key")
        return False
    
    if len(api_key) < 20:
        print("❌ OPENAI_API_KEY appears to be too short")
        print("   Check that you copied the full API key")
        return False
    
    print(f"✅ OpenAI API key is set (starts with: {api_key[:8]}...)")
    return True


def test_imports():
    """Test that all required packages can be imported."""
    print("📦 Testing package imports...")
    
    try:
        import graphiti_core
        print(f"✅ graphiti-core imported successfully (version: {graphiti_core.__version__ if hasattr(graphiti_core, '__version__') else 'unknown'})")
    except ImportError as e:
        print(f"❌ Failed to import graphiti-core: {e}")
        return False
    
    try:
        import openai
        print(f"✅ openai package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import openai: {e}")
        return False
    
    try:
        from graphiti_core import Graphiti
        print("✅ Graphiti class imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Graphiti class: {e}")
        return False
    
    return True


async def main():
    """Run all connection and setup tests."""
    print("🚀 Graphiti Neo4j Setup Test")
    print("=" * 40)
    print()
    
    # Test imports first
    imports_ok = test_imports()
    print()
    
    # Test OpenAI API key
    openai_ok = test_openai_key()
    print()
    
    # Test Neo4j connection
    neo4j_ok = await test_neo4j_connection()
    print()
    
    # Summary
    print("📊 Test Summary")
    print("-" * 20)
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   OpenAI:  {'✅ PASS' if openai_ok else '❌ FAIL'}")
    print(f"   Neo4j:   {'✅ PASS' if neo4j_ok else '❌ FAIL'}")
    print()
    
    if imports_ok and openai_ok and neo4j_ok:
        print("🎉 All tests passed! You're ready to run the quickstart:")
        print("   python quickstart_neo4j.py")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues above before proceeding.")
        print("   See SETUP_INSTRUCTIONS.md for detailed setup guidance.")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main()) 