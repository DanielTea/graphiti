#!/bin/bash

# Graphiti MCP Server Startup Script
echo "🚀 Starting Graphiti MCP Server"
echo "==============================="

# Navigate to MCP server directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Start the server with your preferred settings
echo "📡 Starting MCP server on http://127.0.0.1:8000/sse"
echo "🎯 Group ID: demo-session"
echo "🧠 Custom entities: enabled"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python graphiti_mcp_server.py \
    --transport sse \
    --use-custom-entities \
    --group-id demo-session \
    --host 127.0.0.1 