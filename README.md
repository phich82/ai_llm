# Softwares
- Python 3.8.* (3.8.10)
- Java
- Docker Desktop (latest)
- Visual Studio Code (latest)

# 1. Start Neo4J (database) via docker

    docker-compose up

# 2. Create virtual environment
    python -m venv .venv

# 3. Activate virtual environment
On Linux Terminal:

    source .venv/Scripts/activate

On Windows Terminal:

    .venv\Scripts\activate

# 4. Install packages
    pip install -r requirements.txt
    pip install <package-name>
    pip install <package-name>==<version>

# 5. Prepare environment variables (in .env file)
    OPENAI_API_KEY=s<openai-api-key-here>
    NEO4J_URI=<bolt://localhost:7687>
    NEO4J_USERNAME=<neo4j>
    NEO4J_PASSWORD=<your_password>

# 6. Run application (Inference)
    python enhancing_rag_with_graph.py

Or open file and run each step:

    enhancing_rag_with_graph.ipynb

