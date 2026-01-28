# memorai 🧠

Local-first AI memory system with semantic search. Built in Rust.

Store anything you want to remember, search it semantically, and generate user profiles — all running locally on your machine. No cloud, no subscriptions, no data leaving your box.

## Features

- **Semantic Search** — Find memories by meaning, not just keywords
- **REST API** — Full HTTP API for integration with any app
- **CLI** — Command-line interface for quick access
- **Profile Builder** — Auto-generate user profiles from stored memories
- **Bulk Import** — Import many memories at once
- **Embedded Database** — SurrealDB runs inside the binary, no separate server
- **Local Embeddings** — Uses Ollama for embeddings, everything stays on your machine

## Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Ollama](https://ollama.com/) with an embedding model

```bash
# Install Ollama and pull the embedding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mxbai-embed-large

# Optional: pull a chat model for profile generation
ollama pull qwen2.5:14b
```

## Install

```bash
git clone https://github.com/iamngoni/memorai.git
cd memorai
cargo build --release

# The binary is at ./target/release/memorai
# Optionally, copy it to your PATH:
sudo cp target/release/memorai /usr/local/bin/
```

## Usage

### Start the API server

```bash
memorai serve
```

Server starts at `http://localhost:8484` by default.

### CLI Commands

```bash
# Add a memory
memorai add "Rust is my favorite programming language" --tags "tech,preferences" --source "conversation"

# Search memories
memorai search "what programming languages do I like?" --limit 5

# View stats
memorai stats

# Generate a profile
memorai profile
```

### Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORAI_PORT` | `8484` | API server port |
| `MEMORAI_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `MEMORAI_EMBED_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `MEMORAI_CHAT_MODEL` | `qwen2.5:14b` | Ollama chat model (for profiles) |
| `MEMORAI_DATA_DIR` | `~/.memorai/data` | Database storage path |

## API Reference

### Health Check

```bash
curl http://localhost:8484/health
```

### Create a Memory

```bash
curl -X POST http://localhost:8484/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love building CLI tools in Rust",
    "tags": ["rust", "cli", "preferences"],
    "source": "conversation"
  }'
```

### Search Memories

```bash
curl "http://localhost:8484/v1/search?q=programming+languages&limit=5"
```

### List Memories

```bash
# All memories (paginated)
curl "http://localhost:8484/v1/memories?page=1&per_page=20"

# Filter by tag
curl "http://localhost:8484/v1/memories?tag=rust"

# Filter by source
curl "http://localhost:8484/v1/memories?source=conversation"
```

### Delete a Memory

```bash
curl -X DELETE http://localhost:8484/v1/memories/{id}
```

### Bulk Import

```bash
curl -X POST http://localhost:8484/v1/memories/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [
      {"text": "First memory", "tags": ["test"]},
      {"text": "Second memory", "tags": ["test"], "source": "import"}
    ]
  }'
```

### Get Stats

```bash
curl http://localhost:8484/v1/stats
```

### Generate Profile

```bash
curl http://localhost:8484/v1/profile
```

## Architecture

```
memorai
├── Axum HTTP server (REST API)
├── SurrealDB embedded (storage + indexing)
├── Ollama client (embeddings + chat)
└── Cosine similarity (vector search)
```

- **Storage**: SurrealDB in embedded/RocksDB mode — the database lives inside your binary. No external database server needed.
- **Embeddings**: Generated via Ollama's local API using mxbai-embed-large (1024-dim vectors).
- **Search**: Cosine similarity computed in Rust over stored embedding vectors.
- **Profiles**: Generated using Ollama's chat model (qwen2.5:14b by default).

## License

MIT — see [LICENSE](LICENSE)
