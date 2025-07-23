#!/usr/bin/env python3
"""
Simple test script to verify the stdio MCP server works correctly.
This simulates how an MCP client would interact with the server.
"""

import asyncio
import json
import subprocess
import sys
import os

async def test_mcp_server():
    """Test the MCP server with stdio transport."""
    print("🔬 Testing Graphiti MCP Server with stdio transport...")
    
    # Set up environment
    env = os.environ.copy()
    env['GROUP_ID'] = 'test-stdio'
    
    # Start the MCP server process
    cmd = ['uv', 'run', 'python', 'graphiti_mcp_server.py', '--use-custom-entities']
    
    print(f"📡 Starting server: {' '.join(cmd)}")
    print(f"🎯 GROUP_ID: {env.get('GROUP_ID')}")
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    try:
        # Send initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("📤 Sending initialize request...")
        process.stdin.write(json.dumps(initialize_request) + '\n')
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"📥 Received response: {response}")
            
            if 'result' in response:
                print("✅ MCP server initialized successfully!")
                print(f"📊 Server capabilities: {response['result'].get('capabilities', {})}")
                return True
            else:
                print("❌ Initialization failed")
                return False
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1) 