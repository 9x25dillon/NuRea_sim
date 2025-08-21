
using CSV, DataFrames, LibPQ, JSON3, UUIDs

const DBURL = get(ENV, "DATABASE_URL", "postgres://chaos_user:chaos_pass@localhost:5432/chaos")

function as_pgvector_str(v::Vector{Float64})
    # Format as pgvector literal: [1.0,2.0,...]
    return "[" * join(string.(v), ",") * "]"
end

function insert_augmented_csv(conn::LibPQ.Connection, path::String; source::String)
    df = CSV.File(path) |> DataFrame
    nrows = size(df, 1)
    ndims = size(df, 2)
    println("Inserting $nrows rows from $(path) with dim=$(ndims)")

    # Prepare statement once
    stmt = LibPQ.prepare(conn, "ins_hd_node",
        "INSERT INTO hd_nodes (id,label,payload,coords,unitary_tag,embedding) " *
        "VALUES ($1,$2,$3,$4,$5,$6::vector) " *
        "ON CONFLICT (id) DO NOTHING"
    )

    for i in 1:nrows
        id = string(uuid4())
        payload = JSON3.write(JSON3.Object("source"=>source, "row_index"=>i))
        coords = [0.0, 0.0, 0.0]
        # Extract row as Float64 vector
        v = Vector{Float64}(df[i, :])  # collects row values
        vstr = as_pgvector_str(v)
        LibPQ.execute(conn, stmt, (id, "quantum_node", payload, coords, "quantum", vstr))
        if i % 1000 == 0
            println("  inserted $i / $nrows ...")
        end
    end
    println("Done: $(path)")
end

function main()
    conn = LibPQ.Connection(DBURL)
    try
        insert_augmented_csv(conn, "raw_augmented.csv"; source="raw_augmented.csv")
        insert_augmented_csv(conn, "test_augmented.csv"; source="test_augmented.csv")
    finally
        close(conn)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
