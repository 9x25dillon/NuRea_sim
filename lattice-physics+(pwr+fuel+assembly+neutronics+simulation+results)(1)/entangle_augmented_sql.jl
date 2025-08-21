
using LibPQ, JSON3

const DBURL = get(ENV, "DATABASE_URL", "postgres://chaos_user:chaos_pass@localhost:5432/chaos")

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

function insert_edges_sql(conn::LibPQ.Connection, pairs, nesting_level::Int, weight::Float64, attrs)
    stmt = LibPQ.prepare(conn, "ins_edge",
        "INSERT INTO hd_edges (src,dst,weight,nesting_level,attrs) " *
        "VALUES ($1,$2,$3,$4,$5) " *
        "ON CONFLICT (src,dst) DO UPDATE SET weight=EXCLUDED.weight, nesting_level=EXCLUDED.nesting_level, attrs=EXCLUDED.attrs"
    )
    tx = LibPQ.Transaction(conn)
    try
        for (s,d) in pairs
            LibPQ.execute(conn, stmt, (s, d, weight, nesting_level, JSON3.write(attrs)))
        end
        LibPQ.commit(tx)
    catch e
        LibPQ.rollback(tx)
        rethrow(e)
    end
end

function entangle_source(conn::LibPQ.Connection, source::String)
    println("Fetching ids for source=$(source) ...")
    ids = fetch_ids(conn, source)
    println("  got $(length(ids)) ids")

    if length(ids) < 2
        println("Not enough nodes to entangle for $(source)")
        return
    end

    p1 = [(ids[i], ids[i+1]) for i in 1:length(ids)-1]
    p5 = [(ids[i], ids[i+5]) for i in 1:length(ids)-5]

    println("Inserting edges for $(source) via SQL ...")
    insert_edges_sql(conn, p1, 0, 1.0, JSON3.Object("type"=>"temporal_step1"))
    if !isempty(p5)
        insert_edges_sql(conn, p5, 1, 0.6, JSON3.Object("type"=>"temporal_step5"))
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
