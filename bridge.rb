#!/usr/bin/env ruby

require 'sinatra'
require 'faraday'
require 'json'
require 'dotenv'
require 'webrick'

# Load environment variables
Dotenv.load

# Configuration
CHAOS_URL = ENV.fetch('CHAOS_URL', 'http://localhost:8081')
DATABASE_URL = ENV.fetch('DATABASE_URL', 'postgres://chaos_user:chaos_pass@localhost:5432/chaos')
PORT = ENV.fetch('PORT', 8085)

# Initialize Faraday client for HTTP requests
conn = Faraday.new(url: CHAOS_URL) do |f|
  f.request :json
  f.response :json
  f.adapter Faraday.default_adapter
end

# Health check endpoint
get '/health' do
  content_type :json
  { status: 'healthy', service: 'ruby_bridge', port: PORT, chaos_url: CHAOS_URL }.to_json
end

# Proxy to ChaosRAGJulia health endpoint
get '/chaos/health' do
  begin
    response = conn.get('/health')
    status response.status
    response.body
  rescue => e
    status 500
    { error: "Failed to connect to ChaosRAGJulia: #{e.message}" }.to_json
  end
end

# Proxy to ChaosRAGJulia vector stats
get '/chaos/vector/stats' do
  begin
    response = conn.get('/chaos/vector/stats')
    status response.status
    response.body
  rescue => e
    status 500
    { error: "Failed to get vector stats: #{e.message}" }.to_json
  end
end

# Enhanced document indexing with validation
post '/chaos/rag/index' do
  begin
    request_payload = JSON.parse(request.body.read)
    
    # Validate required fields
    unless request_payload['docs'] && request_payload['docs'].is_a?(Array)
      return [400, { error: 'Missing or invalid docs array' }.to_json]
    end
    
    # Forward to ChaosRAGJulia
    response = conn.post('/chaos/rag/index') do |req|
      req.body = request_payload.to_json
    end
    
    status response.status
    response.body
  rescue JSON::ParserError => e
    status 400
    { error: "Invalid JSON: #{e.message}" }.to_json
  rescue => e
    status 500
    { error: "Indexing failed: #{e.message}" }.to_json
  end
end

# Batch vector similarity search
post '/chaos/rag/query/batch' do
  begin
    request_payload = JSON.parse(request.body.read)
    queries = request_payload['queries'] || []
    
    if queries.empty?
      return [400, { error: 'No queries provided' }.to_json]
    end
    
    results = []
    queries.each do |query|
      begin
        response = conn.post('/chaos/rag/query') do |req|
          req.body = { q: query['text'], k: query['k'] || 12 }.to_json
        end
        
        if response.success?
          results << { query: query['text'], result: response.body, status: 'success' }
        else
          results << { query: query['text'], error: response.body, status: 'failed' }
        end
      rescue => e
        results << { query: query['text'], error: e.message, status: 'error' }
      end
    end
    
    content_type :json
    { results: results, total_queries: queries.length }.to_json
  rescue JSON::ParserError => e
    status 400
    { error: "Invalid JSON: #{e.message}" }.to_json
  rescue => e
    status 500
    { error: "Batch query failed: #{e.message}" }.to_json
  end
end

# Data ingestion bridge for lattice physics data
post '/chaos/ingest/lattice_physics' do
  begin
    request_payload = JSON.parse(request.body.read)
    source = request_payload['source']
    vectors = request_payload['vectors'] || []
    
    if vectors.empty?
      return [400, { error: 'No vectors provided' }]
    end
    
    # Convert to ChaosRAGJulia format
    docs = vectors.map.with_index do |vector, index|
      {
        source: source,
        kind: 'lattice_physics_vector',
        content: "Lattice physics simulation vector #{index + 1}",
        meta: {
          source: source,
          row_index: index + 1,
          dimensions: vector.length,
          timestamp: Time.now.iso8601
        }
      }
    end
    
    # Forward to ChaosRAGJulia
    response = conn.post('/chaos/rag/index') do |req|
      req.body = { docs: docs }.to_json
    end
    
    status response.status
    response.body
  rescue JSON::ParserError => e
    status 400
    { error: "Invalid JSON: #{e.message}" }.to_json
  rescue => e
    status 500
    { error: "Lattice physics ingestion failed: #{e.message}" }.to_json
  end
end

# Temporal edge creation for lattice physics data
post '/chaos/graph/temporal_edges' do
  begin
    request_payload = JSON.parse(request.body.read)
    source = request_payload['source']
    step1_weight = request_payload['step1_weight'] || 1.0
    step5_weight = request_payload['step5_weight'] || 0.6
    
    # Get node IDs ordered by row_index
    response = conn.get("/chaos/graph/nodes_by_source?source=#{source}")
    
    unless response.success?
      return [500, { error: "Failed to get nodes for source: #{source}" }.to_json]
    end
    
    nodes = JSON.parse(response.body)['nodes']
    
    if nodes.length < 2
      return [400, { error: "Not enough nodes for temporal edges" }.to_json]
    end
    
    # Create step-1 edges (i -> i+1)
    step1_pairs = []
    (0...nodes.length-1).each do |i|
      step1_pairs << [nodes[i]['id'], nodes[i+1]['id']]
    end
    
    # Create step-5 edges (i -> i+5)
    step5_pairs = []
    (0...nodes.length-5).each do |i|
      step5_pairs << [nodes[i]['id'], nodes[i+5]['id']]
    end
    
    # Post step-1 edges
    step1_response = conn.post('/chaos/graph/entangle') do |req|
      req.body = {
        pairs: step1_pairs,
        nesting_level: 0,
        weight: step1_weight,
        attrs: { type: 'temporal_step1' }
      }.to_json
    end
    
    # Post step-5 edges
    step5_response = conn.post('/chaos/graph/entangle') do |req|
      req.body = {
        pairs: step5_pairs,
        nesting_level: 1,
        weight: step5_weight,
        attrs: { type: 'temporal_step5' }
      }.to_json
    end
    
    content_type :json
    {
      source: source,
      step1_edges: { count: step1_pairs.length, status: step1_response.status },
      step5_edges: { count: step5_pairs.length, status: step5_response.status },
      total_nodes: nodes.length
    }.to_json
  rescue => e
    status 500
    { error: "Temporal edge creation failed: #{e.message}" }.to_json
  end
end

# Get nodes by source (ordered by row_index)
get '/chaos/graph/nodes_by_source' do
  begin
    source = params['source']
    
    # This would need to be implemented in ChaosRAGJulia
    # For now, return a placeholder
    content_type :json
    { 
      source: source,
      nodes: [],
      message: "This endpoint needs to be implemented in ChaosRAGJulia"
    }.to_json
  rescue => e
    status 500
    { error: "Failed to get nodes by source: #{e.message}" }.to_json
  end
end

# Root endpoint
get '/' do
  content_type :json
  {
    service: 'ChaosRAGJulia Ruby Bridge',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      chaos_health: '/chaos/health',
      vector_stats: '/chaos/vector/stats',
      index_docs: '/chaos/rag/index',
      batch_query: '/chaos/rag/query/batch',
      lattice_physics_ingest: '/chaos/ingest/lattice_physics',
      temporal_edges: '/chaos/graph/temporal_edges',
      nodes_by_source: '/chaos/graph/nodes_by_source'
    },
    environment: {
      chaos_url: CHAOS_URL,
      database_url: DATABASE_URL,
      port: PORT
    }
  }.to_json
end

# Start the server
puts "🚀 Starting ChaosRAGJulia Ruby Bridge on port #{PORT}"
puts "📡 Connecting to ChaosRAGJulia at: #{CHAOS_URL}"
puts "🗄️  Database: #{DATABASE_URL}"
puts "🌐 Available at: http://localhost:#{PORT}"

set :port, PORT
set :bind, '0.0.0.0'
