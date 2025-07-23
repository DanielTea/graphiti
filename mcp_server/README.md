# Graphiti MCP Server (FalkorDB)

A Model Context Protocol (MCP) server that provides memory capabilities using Graphiti with FalkorDB as the graph database backend.

## Features

- **Graph-based memory**: Store and retrieve information using Graphiti's knowledge graph
- **FalkorDB backend**: High-performance Redis-compatible graph database
- **Entity extraction**: Automatically extract entities and relationships from text
- **Multi-tenancy**: Separate databases per group_id for data isolation
- **stdio transport**: Compatible with MCP clients using command-based configuration

## Installation

```bash
cd mcp_server
uv sync
```

## Usage

### With MCP Client Configuration

Add this configuration to your MCP client (e.g., Claude Desktop):

```json
{
  "name": "graphiti_mcp_server",
  "command": "uv",
  "args": ["run", "python", "graphiti_mcp_server.py", "--use-custom-entities"],
  "cwd": "/path/to/graphiti/mcp_server",
  "env": {
    "GROUP_ID": "your-session-id"
  }
}
```

### Environment Variables

- `GROUP_ID` or `group_id`: Unique identifier for your session/database
- `OPENAI_API_KEY`: OpenAI API key for LLM operations
- `FALKORDB_HOST`: FalkorDB host (default: localhost)
- `FALKORDB_PORT`: FalkorDB port (default: 6379)

### Testing

Test the server locally:

```bash
cd mcp_server
GROUP_ID=test123 uv run python graphiti_mcp_server.py --use-custom-entities
```

## Available Tools

- `add_memory`: Add new information to the graph
- `search_memory_nodes`: Search for relevant nodes/entities
- `search_memory_facts`: Search for relevant facts/relationships
- `get_episodes`: Get recent memory episodes
- `clear_graph`: Clear all data for a group
- `get_entity_edge`: Get specific entity relationships
- `delete_episode`: Delete specific episodes
- `delete_entity_edge`: Delete specific relationships

## Database Management

Each `GROUP_ID` creates a separate FalkorDB database, enabling multi-tenancy and data isolation. Databases are created automatically when first accessed.

## Prerequisites

- FalkorDB server running (see main project README for setup)
- OpenAI API key
- Python 3.10+
