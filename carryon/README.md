# Kiln Orchestrator

Kiln is a Ruby-based orchestrator that manages your Suno2OpenAI ecosystem services including Chaos RAG (Julia), Suno API (Python FastAPI), and Desktop Electron app.

## Architecture

- **Chaos RAG**: Julia server on port 8081 with chaos RAG endpoints
- **Suno API**: Python FastAPI service on port 8001 for cookies management and chat
- **Desktop App**: Electron app with development server on port 5173
- **PostgreSQL**: Database for chaos RAG and Suno API

## Quick Start

### 1. Install Dependencies

```bash
cd kiln
bundle install
```

### 2. Start Services

```bash
# Start chaos RAG (Julia server)
./bin/kiln service start chaos_rag

# Start Suno API (Python FastAPI)
./bin/kiln service start suno_api

# Start desktop app (Electron)
./bin/kiln service start desktop_app
```

### 3. Check Service Health

```bash
./bin/kiln service health chaos_rag
./bin/kiln service health suno_api
./bin/kiln service health desktop_app
```

## Usage Examples

### Chaos RAG Operations

```bash
# Query chaos RAG
./bin/kiln chaos query "What's the current ETH volatility regime?"

# Ingest telemetry data
./bin/kiln chaos telemetry ETH 0.15 0.8

# Ingest HHT data
echo '{"x":[1.0,2.0,3.0],"ts":["2024-01-01T00:00:00Z","2024-01-01T00:01:00Z","2024-01-01T00:02:00Z"]}' > hht_data.json
./bin/kiln chaos hht ETH hht_data.json

# Check health
./bin/kiln chaos health
```

### Suno API Operations

```bash
# Set auth key
export SUNO_AUTH_KEY="your_actual_auth_key"

# List cookies
./bin/kiln suno cookies list

# Add cookies
echo '{"cookies":["cookie1","cookie2"]}' > cookies.json
./bin/kiln suno cookies add cookies.json

# Refresh cookies
./bin/kiln suno cookies refresh

# Clean invalid cookies
./bin/kiln suno cookies clean

# Chat completion
echo '{"messages":[{"role":"user","content":"Generate a jazz melody"}]}' > chat.json
./bin/kiln suno chat chat.json

# Check health
./bin/kiln suno health
```

### Desktop App Operations

```bash
# Start development server
./bin/kiln desktop dev

# Build for production
./bin/kiln desktop build
```

### Matrix Optimization

```bash
# Create matrix data
echo '{"matrix":[[1.0,0.5],[0.2,0.3]]}' > matrix.json

# Run optimization
./bin/kiln optimize --json matrix.json --method entropy --target-entropy 0.6
```

## Docker Compose

Start the entire stack:

```bash
docker compose up -d --build
```

This will start:
- PostgreSQL database
- Chaos RAG Julia server
- Suno API Python service  
- Desktop Electron app
- Kiln orchestrator

## Service Endpoints

### Chaos RAG (Julia - Port 8081)
- `POST /chaos/rag/query` - RAG query with context routing
- `POST /chaos/telemetry` - Ingest market telemetry
- `POST /chaos/hht/ingest` - Ingest HHT analysis data
- `POST /chaos/graph/entangle` - Create graph relationships
- `GET /chaos/graph/{id}` - Get graph node details

### Suno API (Python - Port 8001)
- `GET /cookies` - List cookies with counts
- `PUT /cookies` - Add new cookies
- `DELETE /cookies` - Delete cookies
- `GET /refresh/cookies` - Refresh existing cookies
- `DELETE /refresh/cookies` - Clean invalid cookies
- `DELETE /songID/cookies` - Update song IDs
- `POST /v1/chat/completions` - Chat completion endpoint

## Configuration

Edit `kiln.yml` to customize:
- Service repository paths
- Health check endpoints
- Artifact storage location

## Artifacts

All command outputs are captured in `.kiln/runs/<timestamp>_<label>/`:
- `stdout.log` - Standard output
- `stderr.log` - Standard error
- `meta.json` - Execution metadata
- Request/response files for API calls

## Development

### Adding New Services

1. Add service definition to `kiln.yml`
2. Create client in `lib/clients/`
3. Add CLI commands in `lib/cli.rb`

### Health Checks

Supported health check types:
- `http` - HTTP endpoint check
- `tcp` - TCP port check
- `file` - File existence check
- `command` - Command execution check

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check if services are already running on required ports
2. **Missing dependencies**: Ensure Julia/Python/Node dependencies are installed
3. **Database connection**: Verify PostgreSQL is running and accessible
4. **Auth keys**: Set `SUNO_AUTH_KEY` environment variable for Suno API

### Logs

Check service logs in `.kiln/runs/` directory for detailed execution information.

## License

MIT License - see LICENSE file for details.
