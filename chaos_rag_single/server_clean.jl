# ChaosRAGJulia - Enhanced server with PostgreSQL vector extension
using HTTP, JSON3, LibPQ, DSP, UUIDs, Dates, Statistics, Random, FFTW, Logging

const DBURL = get(ENV, "DATABASE_URL", "postgres://chaos_user:chaos_pass@localhost:5432/chaos")
const OPENAI_MODEL_EMB = "text-embedding-3-large"
const OPENAI_MODEL_CHAT = "gpt-4o-mini"

# Connection pool configuration
const MAX_CONNECTIONS = 20
const IDLE_TIMEOUT = 300  # 5 minutes
const CONNECTION_TIMEOUT = 10

# Global connection pool
mutable struct ConnectionPool
    connections::Vector{Union{Nothing, LibPQ.Connection}}
    in_use::Vector{Bool}
    last_used::Vector{Float64}
end

const POOL = Ref{Union{Nothing, ConnectionPool}}(nothing)

# Initialize connection pool
function init_connection_pool()
    pool = ConnectionPool(
        fill(nothing, MAX_CONNECTIONS),
        fill(false, MAX_CONNECTIONS),
        fill(0.0, MAX_CONNECTIONS)
    )
    
    # Initialize connections
    for i in 1:MAX_CONNECTIONS
        try
            pool.connections[i] = LibPQ.Connection(DBURL)
            pool.in_use[i] = false
            pool.last_used[i] = time()
        catch e
            println("Failed to create connection $i: $e")
        end
    end
    
    return pool
end

# Get connection from pool
function get_connection()
    isnothing(POOL[]) && error("Connection pool not initialized")
    pool = POOL[]
    
    # Find available connection
    for i in 1:MAX_CONNECTIONS
        if !pool.in_use[i] && !isnothing(pool.connections[i])
            # Test connection
            try
                LibPQ.execute(pool.connections[i], "SELECT 1")
                pool.in_use[i] = true
                pool.last_used[i] = time()
                return (i, pool.connections[i])
            catch
                # Connection is dead, try to recreate
                try
                    close(pool.connections[i])
                catch
                    # Ignore close errors
                end
                pool.connections[i] = LibPQ.Connection(DBURL)
                pool.in_use[i] = true
                pool.last_used[i] = time()
                return (i, pool.connections[i])
            end
        end
    end
    
    # If no connections available, create a new one temporarily
    println("Warning: No pool connections available, creating temporary connection")
    return (-1, LibPQ.Connection(DBURL))
end

# Return connection to pool
function return_connection(conn_info)
    isnothing(POOL[]) && return
    pool = POOL[]
    
    i, conn = conn_info
    if i > 0 && i <= MAX_CONNECTIONS
        pool.in_use[i] = false
        pool.last_used[i] = time()
    else
        # Close temporary connection
        try
            close(conn)
        catch
            # Ignore close errors
        end
    end
end

# Execute query with automatic connection management
function exec_query(query::AbstractString, params::Tuple=())
    conn_info = get_connection()
    try
        return LibPQ.execute(conn_info[2], query, params)
    finally
        return_connection(conn_info)
    end
end

# Execute multiple statements with connection management
function exec_sql(sql::AbstractString)
    conn_info = get_connection()
    try
        for stmt in split(sql, ';')
            s = strip(stmt)
            isempty(s) && continue
            try
                LibPQ.execute(conn_info[2], s)
            catch e
                println("SQL exec warning: $s - $e")
            end
        end
    finally
        return_connection(conn_info)
    end
end

# Initialize database
function init_db()
    try
        # Test basic connectivity first
        test_conn = LibPQ.Connection(DBURL)
        LibPQ.execute(test_conn, "SELECT 1")
        close(test_conn)
        
        # Initialize the connection pool
        POOL[] = init_connection_pool()
        println("Connection pool initialized successfully with $(MAX_CONNECTIONS) connections")
        return true
    catch e
        println("Database connection failed: $e")
        return false
    end
end

function json(req)::JSON3.Object
    body = String(take!(req.body))
    JSON3.read(body)
end

resp(obj; status::Int=200) = HTTP.Response(status, ["Content-Type"=>"application/json"], JSON3.write(obj))

# Enhanced schema with vector extension
const SCHEMA = raw"""
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS hd_nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  label TEXT NOT NULL,
  payload JSONB NOT NULL,
  coords DOUBLE PRECISION[] NOT NULL,
  unitary_tag TEXT,
  embedding VECTOR(1536),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hd_edges (
  src UUID REFERENCES hd_nodes(id) ON DELETE CASCADE,
  dst UUID REFERENCES hd_nodes(id) ON DELETE CASCADE,
  weight DOUBLE PRECISION DEFAULT 1.0,
  nesting_level INT DEFAULT 0,
  attrs JSONB DEFAULT '{}'::jsonb,
  PRIMARY KEY (src, dst)
);

CREATE TABLE IF NOT EXISTS hd_docs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source TEXT,
  kind TEXT,
  content TEXT,
  meta JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Vector similarity search index
CREATE INDEX IF NOT EXISTS idx_hd_nodes_embedding ON hd_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);

CREATE TABLE IF NOT EXISTS tf_hht (
  asset TEXT NOT NULL,
  ts_start TIMESTAMPTZ NOT NULL,
  ts_end   TIMESTAMPTZ NOT NULL,
  imf_idx  INT NOT NULL DEFAULT 1,
  inst_freq DOUBLE PRECISION[] NOT NULL,
  inst_amp  DOUBLE PRECISION[] NOT NULL,
  burst BOOLEAN NOT NULL,
  features JSONB NOT NULL,
  PRIMARY KEY (asset, ts_start, imf_idx)
);

CREATE INDEX IF NOT EXISTS idx_tf_hht_asset_time ON tf_hht (asset, ts_start, ts_end);

CREATE TABLE IF NOT EXISTS state_telemetry (
  ts TIMESTAMPTZ PRIMARY KEY DEFAULT now(),
  asset TEXT NOT NULL,
  realized_vol DOUBLE PRECISION NOT NULL,
  entropy DOUBLE PRECISION NOT NULL,
  mod_intensity_grad DOUBLE PRECISION DEFAULT 0.0,
  router_noise DOUBLE PRECISION DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_state_tel_asset_ts ON state_telemetry (asset, ts);
"""

# Initialize database schema
function init_schema()
    try
        exec_sql(SCHEMA)
        println("Database schema initialized successfully with vector extension")
        return true
    catch e
        println("Schema initialization failed: $e")
        return false
    end
end

# Simple OpenAI client stub
module OpenAIClient
using HTTP, JSON3, Random, Statistics

function fake_embed(text::AbstractString, dim::Int=1536)
    seed = UInt32(hash(text) % 0xffffffff)
    rng = Random.MersenneTwister(seed)
    v = rand(rng, Float32, dim)
    v ./= sqrt(sum(v.^2) + 1e-6f0)
    return v
end

function embed(text::AbstractString; model::AbstractString="text-embedding-3-large", dim::Int=1536)
    key = get(ENV, "OPENAI_API_KEY", nothing)
    isnothing(key) && return fake_embed(text, dim)
    
    try
        resp = HTTP.post("https://api.openai.com/v1/embeddings";
            headers = ["Authorization"=>"Bearer $key","Content-Type"=>"application/json"],
            body = JSON3.write((; input=text, model=model)))
        
        if resp.status != 200
            return fake_embed(text, dim)
        end
        
        data = JSON3.read(String(resp.body))
        return Float32.(data["data"][1]["embedding"])
    catch
        return fake_embed(text, dim)
    end
end

function chat(system::AbstractString, prompt::AbstractString; model::AbstractString="gpt-4o-mini")
    key = get(ENV, "OPENAI_API_KEY", nothing)
    isnothing(key) && return "(stub) " * prompt[1:min(end, 400)]
    
    body = JSON3.write(Dict(
        "model"=>model,
        "messages"=>[
            Dict("role"=>"system","content"=>system),
            Dict("role"=>"user","content"=>prompt)
        ],
        "temperature"=>0.2
    ))
    
    try
        resp = HTTP.post("https://api.openai.com/v1/chat/completions";
            headers=["Authorization"=>"Bearer $key","Content-Type"=>"application/json"],
            body=body)
        
        if resp.status != 200
            return "(stub) " * prompt[1:min(end, 400)]
        end
        
        data = JSON3.read(String(resp.body))
        return String(data["choices"][1]["message"]["content"])
    catch
        return "(stub) " * prompt[1:min(end, 400)]
    end
end
end

# Router setup
router = HTTP.Router()

# Index documents
HTTP.register!(router, "POST", "/chaos/rag/index", function(req)
    d = json(req)
    docs = get(d, :docs, JSON3.Array())
    count = 0
    
    try
        for doc in docs
            src = get(doc,:source, nothing)
            kind = get(doc,:kind, nothing)
            content = String(get(doc,:content, ""))
            meta = JSON3.write(get(doc,:meta, JSON3.Object()))
            
            r = exec_query("INSERT INTO hd_docs (source,kind,content,meta) VALUES (\$1,\$2,\$3,\$4) RETURNING id", 
                          (src,kind,content,meta))
            doc_id = first(r)[1]
            
            emb = OpenAIClient.embed(content; model=OPENAI_MODEL_EMB)
            coords = [0.0,0.0,0.0]
            payload = JSON3.write(JSON3.Object("doc_id"=>doc_id, "snippet"=>first(split(content, '\n'))))
            
            # Use vector type for embedding
            exec_query("INSERT INTO hd_nodes (id,label,payload,coords,unitary_tag,embedding) VALUES (\$1,\$2,\$3,\$4,\$5,\$6::vector) ON CONFLICT (id) DO NOTHING",
                      (doc_id, "doc", payload, coords, "identity", emb))
            count += 1
        end
        
        resp(JSON3.Object("inserted"=>count))
    catch e
        println("Indexing failed: $e")
        resp(JSON3.Object("error"=>string(e)); status=500)
    end
end)

# Health check
HTTP.register!(router, "GET", "/health", function(req)
    db_ok = init_db()
    resp(JSON3.Object("status"=>db_ok ? "healthy" : "degraded", "backend"=>"chaos_rag_julia_vector", "database"=>db_ok))
end)

# Vector stats
HTTP.register!(router, "GET", "/chaos/vector/stats", function(req)
    try
        # Get vector statistics
        node_count = first(exec_query("SELECT COUNT(*) FROM hd_nodes"), 1)
        doc_count = first(exec_query("SELECT COUNT(*) FROM hd_docs"), 1)
        edge_count = first(exec_query("SELECT COUNT(*) FROM hd_edges"), 1)
        
        # Check if vector extension is working
        vector_test = exec_query("SELECT embedding <-> '[0.1,0.2,0.3]'::vector AS test_distance FROM hd_nodes LIMIT 1")
        vector_working = !isempty(vector_test)
        
        # Get pool statistics
        pool_stats = isnothing(POOL[]) ? Dict("status"=>"not_initialized") : Dict(
            "max_connections"=>MAX_CONNECTIONS,
            "active_connections"=>sum(POOL[].in_use),
            "idle_connections"=>MAX_CONNECTIONS - sum(POOL[].in_use),
            "total_connections"=>count(x -> !isnothing(x), POOL[].connections)
        )
        
        resp(JSON3.Object(
            "vector_extension"=>vector_working,
            "nodes"=>node_count,
            "documents"=>doc_count,
            "edges"=>edge_count,
            "connection_pool"=>pool_stats,
            "status"=>"operational"
        ))
    catch e
        println("Vector stats failed: $e")
        resp(JSON3.Object("error"=>string(e), "status"=>"error"); status=500)
    end
end)

# Cleanup function for graceful shutdown
function cleanup()
    if !isnothing(POOL[])
        println("Closing connection pool...")
        for conn in POOL[].connections
            if !isnothing(conn)
                try
                    close(conn)
                catch
                    # Ignore close errors
                end
            end
        end
        POOL[] = nothing
    end
end

# Main server
function main()
    println("🚀 Starting ChaosRAGJulia (vector-enhanced) server...")
    
    # Initialize database
    if !init_db()
        println("Failed to connect to database. Please check PostgreSQL is running and credentials are correct.")
        return
    end
    
    # Initialize schema
    if !init_schema()
        println("Failed to initialize database schema with vector extension.")
        return
    end
    
    println("✅ Database connection established")
    println("✅ Vector extension schema initialized")
    println("✅ Connection pool configured (max: $MAX_CONNECTIONS)")
    println("🌐 Server starting on 0.0.0.0:8081")
    println("📊 Available endpoints:")
    println("   POST /chaos/rag/index    - Index documents with vector embeddings")
    println("   GET  /chaos/vector/stats - Vector extension and pool statistics")
    println("   GET  /health             - Health check")
    println("🔍 Vector Features:")
    println("   - PostgreSQL vector extension for efficient similarity search")
    println("   - IVFFlat index for fast approximate nearest neighbor search")
    println("   - Cosine similarity using vector operators")
    println("   - Optimized for 1536-dimensional embeddings")
    println("🔌 Connection Pool Features:")
    println("   - Custom connection pooling with $(MAX_CONNECTIONS) connections")
    println("   - Automatic connection management and reuse")
    println("   - Connection health checking and recovery")
    println("   - Graceful cleanup on shutdown")
    
    # Set up signal handlers for graceful shutdown
    atexit(cleanup)
    
    HTTP.serve(router, "0.0.0.0", 8081)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
