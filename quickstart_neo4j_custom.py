#!/usr/bin/env python3
"""
Graphiti Neo4j Quickstart - Customized for your local setup

This is a customized version of the quickstart with your specific Neo4j settings:
- URI: neo4j://127.0.0.1:7687
- Database: graphiti
- Password: (as configured)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Your specific Neo4j connection parameters
neo4j_uri = 'neo4j://127.0.0.1:7687'
neo4j_user = 'neo4j'
neo4j_password = '12345678'
neo4j_database = 'graphiti'

print(f"🔗 Connecting to Neo4j:")
print(f"   URI: {neo4j_uri}")
print(f"   Database: {neo4j_database}")
print(f"   User: {neo4j_user}")
print()

async def main():
    # Check for OpenAI API key
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key or openai_key in ['your_openai_api_key_here', 'your_actual_openai_api_key']:
        print("❌ OpenAI API key is not set or is a placeholder value.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_actual_api_key")
        print()
        print("You can get an API key from: https://platform.openai.com/api-keys")
        return

    print(f"✅ OpenAI API key configured (starts with: {openai_key[:12]}...)")
    print()

    # Initialize Graphiti with your specific Neo4j settings
    graphiti = Graphiti(
        uri=neo4j_uri, 
        user=neo4j_user, 
        password=neo4j_password,
        database=neo4j_database
    )

    try:
        print("🔧 Initializing Graphiti indices and constraints...")
        await graphiti.build_indices_and_constraints()
        print("✅ Graph database initialized successfully!")
        print()

        # Example episodes about your setup
        episodes = [
            {
                'content': 'I successfully set up Graphiti with my local Neo4j database. '
                'The connection uses neo4j://127.0.0.1:7687 and connects to a database named graphiti.',
                'type': EpisodeType.text,
                'description': 'setup documentation',
            },
            {
                'content': 'Graphiti is a framework for building temporally-aware knowledge graphs. '
                'It can integrate user interactions and enterprise data into a queryable graph.',
                'type': EpisodeType.text,
                'description': 'system documentation',
            },
            {
                'content': {
                    'system': 'Graphiti',
                    'database': 'Neo4j',
                    'version': 'Neo4j 5.x',
                    'connection_type': 'local',
                    'features': ['temporal queries', 'semantic search', 'graph reasoning']
                },
                'type': EpisodeType.json,
                'description': 'system metadata',
            },
        ]

        # Add episodes to the graph
        print("📝 Adding episodes to the knowledge graph...")
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Setup Episode {i+1}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'   ✅ Added: Setup Episode {i+1} ({episode["type"].value})')

        print()

        # Demonstrate search functionality
        print("🔍 Testing search functionality...")
        print("Query: 'How do I set up Graphiti with Neo4j?'")
        results = await graphiti.search('How do I set up Graphiti with Neo4j?')

        print(f"\n📊 Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f'\n{i}. Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'   Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'   Valid until: {result.invalid_at}')

        # Test node search
        if results:
            center_node_uuid = results[0].source_node_uuid
            print(f"\n🎯 Testing graph-aware search with center node: {center_node_uuid[:8]}...")
            
            reranked_results = await graphiti.search(
                'What is Graphiti?', 
                center_node_uuid=center_node_uuid
            )

            print(f"📊 Reranked results: {len(reranked_results)} found")
            for i, result in enumerate(reranked_results[:2], 1):  # Show top 2
                print(f'\n{i}. {result.fact}')

        # Node search using recipes
        print(f"\n🔎 Testing node search...")
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 3

        node_results = await graphiti._search(
            query='Graphiti database system',
            config=node_search_config,
        )

        print(f"📊 Node search results: {len(node_results.nodes)} nodes found")
        for i, node in enumerate(node_results.nodes, 1):
            print(f'\n{i}. Node: {node.name}')
            summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'   Summary: {summary}')
            print(f'   Labels: {", ".join(node.labels)}')

        print()
        print("🎉 Quickstart completed successfully!")
        print("💡 Your Neo4j database now contains knowledge about your Graphiti setup.")
        print(f"🌐 You can explore the graph visually at: http://localhost:7474")
        print("📚 Try modifying the episodes above with your own data!")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Quickstart failed: {e}")
    finally:
        await graphiti.close()
        print("\n🔌 Connection closed.")


if __name__ == '__main__':
    asyncio.run(main()) 