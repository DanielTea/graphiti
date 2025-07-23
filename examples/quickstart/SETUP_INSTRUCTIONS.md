# Graphiti Neo4j Local Setup Instructions

## ✅ Prerequisites Completed
- [x] Python virtual environment created
- [x] Graphiti-core and dependencies installed
- [x] Neo4j driver (v5.28.1) available

## 🚀 Next Steps to Complete Setup

### 1. Start Your Neo4j Database

You need to have Neo4j running locally. Here are the most common options:

#### Option A: Neo4j Desktop (Recommended)
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database project
3. Create a local DBMS with these settings:
   - **Name**: Any name you prefer
   - **Password**: Set a password (remember it!)
   - **Version**: Neo4j 5.x recommended
4. **Start the database** - this is crucial!
5. Note the connection details (usually \`bolt://localhost:7687\`)

#### Option B: Docker
\`\`\`bash
docker run \\
    --name neo4j \\
    -p7474:7474 -p7687:7687 \\
    -d \\
    -v \$HOME/neo4j/data:/data \\
    -v \$HOME/neo4j/logs:/logs \\
    --env NEO4J_AUTH=neo4j/password \\
    neo4j:latest
\`\`\`

### 2. Configure Environment Variables

Set these environment variables in your shell:

\`\`\`bash
# OpenAI API Key (Required for LLM and embedding functionality)
export OPENAI_API_KEY=your_actual_openai_api_key

# Neo4j Connection Parameters
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_neo4j_password
\`\`\`

### 3. Test Your Connection

\`\`\`bash
# Activate the environment
source .venv/bin/activate

# Test the connection
python test_connection.py
\`\`\`

### 4. Run the Quickstart Example

Once tests pass:

\`\`\`bash
python quickstart_neo4j.py
\`\`\`

## 🔧 Troubleshooting

- **Connection Refused**: Neo4j is not running
- **Authentication Failed**: Wrong username/password  
- **Database Not Found**: Add \`NEO4J_DATABASE=your_database_name\`
- **OpenAI Issues**: Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## 💡 Tips

- Keep Neo4j browser open at http://localhost:7474 to visualize the graph
- Each run adds new data unless you clear the database first
