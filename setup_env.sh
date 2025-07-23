#!/bin/bash

# Graphiti Quickstart Environment Setup
# Source this file to set up your environment variables

echo "Setting up Graphiti environment variables..."

# OpenAI API Key (Required for LLM and embedding functionality)
# IMPORTANT: Replace 'your_openai_api_key_here' with your actual OpenAI API key
export OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Connection Parameters
# These are the default values for a local Neo4j Desktop installation
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

echo "Environment variables set!"
echo "IMPORTANT: Don't forget to replace 'your_openai_api_key_here' with your actual OpenAI API key"
echo ""
echo "To use these settings, run:"
echo "source setup_env.sh"
echo ""
echo "Your current Neo4j connection settings:"
echo "  URI: $NEO4J_URI"
echo "  User: $NEO4J_USER" 
echo "  Password: $NEO4J_PASSWORD" 