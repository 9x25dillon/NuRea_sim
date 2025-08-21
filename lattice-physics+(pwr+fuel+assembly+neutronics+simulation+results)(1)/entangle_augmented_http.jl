
using LibPQ, JSON3, HTTP

const DBURL = get(ENV, "DATABASE_URL", "postgres://chaos_user:chaos_pass@localhost:5432/chaos")
const CHAOS_URL = get(ENV, "CHAOS_URL", "http://localhost:8081")

"""
Fetch node IDs ordered by row_index for a given CSV source label stored in payload JSONB.
"""
function fetch_ids(conn::LibPQ.Connection, source::String)
    sql = """
    SELECT id
    FROM hd_nodes
    WHERE payload->>'source' = $1
    ORDER BY (payload->>'row_index')::int ASC
    """
    ids = String[]
    for r in LibPQ.execute(conn, sql, (source,))
        push!(ids, String(r[1]))
    end
    return ids
end

"""
Build edge pairs for temporal links:
- i -> i+1 (weight 1.0, nesting_level 0)
- i -> i+5 (weight 0.6, nesting_level 1), if in bounds
"""
function build_pairs(ids::Vector{String})
    pairs_step1 = [(ids[i], ids[i+1]) for i in 1:length(ids)-1]
    pairs_step5 = [(ids[i], ids[i+5]) for i in 1:length(ids)-5]
    return pairs_step1, pairs_step5
end

function post_pairs(pairs; nesting_level::Int, weight::Float64, attrs=JSON3.Object(), batch_size::Int=1000)
    i = 1
    while i <= length(pairs)
        j = min(i+batch_size-1, length(pairs))
        body = JSON3.write(JSON3.Object(
            "pairs" => [ (p[1], p[2]) for p in pairs[i:j] ],
            "nesting_level" => nesting_level,
            "weight" => weight,
            "attrs" => attrs
        ))
        resp = HTTP.post("$(CHAOS_URL)/chaos/graph/entangle";
            headers = ["Content-Type" => "application/json"],
            body = body)
        if resp.status != 200
            error("Entangle POST failed with status $(resp.status): $(String(resp.body))")
        end
        println("  posted pairs $(i)-$(j) (nesting $(nesting_level), weight $(weight))")
        i = j + 1
    end
end

function entangle_source(conn, source::String)
    println("Fetching ids for source=$(source) ...")
    ids = fetch_ids(conn, source)
    println("  got $(length(ids)) ids")

    if length(ids) < 2
        println("Not enough nodes to entangle for $(source)")
        return
    end

    p1, p5 = build_pairs(ids)
    println("Posting /chaos/graph/entangle for $(source) ...")
    post_pairs(p1; nesting_level=0, weight=1.0, attrs=JSON3.Object("type"=>"temporal_step1"))
    if !isempty(p5)
        post_pairs(p5; nesting_level=1, weight=0.6, attrs=JSON3.Object("type"=>"temporal_step5"))
    end
    println("Done: $(source)")
end

function main()
    conn = LibPQ.Connection(DBURL)
    try
        entangle_source(conn, "raw_augmented.csv")
        entangle_source(conn, "test_augmented.csv")
    finally
        close(conn)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
