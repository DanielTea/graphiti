#!/bin/bash

echo "🚀 Graphiti Neo4j Quickstart Runner"
echo "=================================="
echo

# Check if OpenAI API key is provided as argument
if [ "$1" != "" ]; then
    export OPENAI_API_KEY="$1"
    echo "✅ Using provided OpenAI API key"
else
    echo "Please provide your OpenAI API key as an argument:"
    echo "  ./run_quickstart.sh sk-your-api-key-here"
    echo
    echo "Or set it manually and run the script:"
    echo "  export OPENAI_API_KEY=sk-your-api-key-here"
    echo "  python quickstart_neo4j_custom.py"
    echo
    echo "Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Run the custom quickstart
echo "Starting Graphiti quickstart with your Neo4j settings..."
echo
python quickstart_neo4j_custom.py
