#!/bin/bash

# Graphiti MCP Server Startup Script (stdio)
echo "🚀 Starting Graphiti MCP Server with stdio transport"
echo "======================================================"

# Navigate to MCP server directory
cd "$(dirname "$0")"

# Test the server with uv run (this is how it will be used in production)
echo "📡 Testing MCP server with stdio transport"
echo "🎯 Group ID: can be set with GROUP_ID or group_id env var"
echo "🔗 Database: FalkorDB (Redis-compatible graph database)"
echo "🧠 Custom entities: enabled"
echo ""
echo "This will start the server in stdio mode for testing"
echo "In production, use: uv run graphiti-mcp-server --use-custom-entities"
echo ""

# Set a test group_id for demonstration
export GROUP_ID=test-stdio-session

uv run graphiti-mcp-server --use-custom-entities
