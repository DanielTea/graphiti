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
5. Note the connection details (usually `bolt://localhost:7687`)

#### Option B: Neo4j Community Server
1. Download Neo4j Community Server
2. Start with: `neo4j start`
3. Default connection: `bolt://localhost:7687`

#### Option C: Docker
```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

### 2. Configure Environment Variables

Set these environment variables in your shell:

```bash
# OpenAI API Key (Required for LLM and embedding functionality)
export OPENAI_API_KEY=your_actual_openai_api_key

# Neo4j Connection Parameters
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_neo4j_password
```

**⚠️ Important Notes:**
- Replace `your_actual_openai_api_key` with your real OpenAI API key
- Replace `your_neo4j_password` with the password you set for your Neo4j database
- If you're using different connection details, update the URI accordingly

### 3. Test Your Connection

Activate the virtual environment and test the connection:

```bash
# Activate the environment
source .venv/bin/activate

# Set your environment variables (replace with actual values)
export OPENAI_API_KEY=your_actual_openai_api_key
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_neo4j_password

# Test Neo4j connection
python -c "
from neo4j import AsyncGraphDatabase
import asyncio

async def test():
    try:
        driver = AsyncGraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))
        await driver.execute_query('RETURN 1 as test')
        print('✅ Neo4j connection successful!')
        await driver.close()
    except Exception as e:
        print(f'❌ Connection failed: {e}')

asyncio.run(test())
"
```

### 4. Run the Quickstart Example

Once your connection test passes:

```bash
python quickstart_neo4j.py
```

## 🔧 Troubleshooting

### "Connection Refused" Error
- **Cause**: Neo4j is not running
- **Solution**: Start your Neo4j database (see step 1)

### "Authentication Failed" Error  
- **Cause**: Wrong username/password
- **Solution**: Check your Neo4j credentials and update `NEO4J_PASSWORD`

### "Database Not Found" Error
- **Cause**: Database name mismatch
- **Solution**: Add `NEO4J_DATABASE=your_database_name` to environment variables

### OpenAI API Issues
- **Cause**: Missing or invalid API key
- **Solution**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## 📚 What the Quickstart Does

1. **Connects** to your Neo4j database
2. **Initializes** Graphiti indices and constraints
3. **Adds sample episodes** about California politicians
4. **Demonstrates search capabilities**:
   - Basic semantic search
   - Graph-aware search with center nodes
   - Node search with predefined recipes
5. **Shows results** with facts, relationships, and temporal data

## 🎯 Next Steps

After successful setup:
- Modify the episodes in `quickstart_neo4j.py` with your own data
- Explore different search queries
- Try the advanced examples in other directories
- Check out the main [Graphiti documentation](https://help.getzep.com/graphiti/)

## 💡 Tips

- Keep your Neo4j browser open at http://localhost:7474 to visualize the graph
- The quickstart creates nodes and relationships that you can explore visually
- Each run will add new data unless you clear the database first 