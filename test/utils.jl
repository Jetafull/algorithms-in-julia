using DelimitedFiles

"""Load union-find test data."""
function loadufdata(datapath)
    rawdata = readdlm(datapath)
    nv = Int(rawdata[1, 1])
    edges = Tuple{Int, Int}[]
    for (d1, d2) in eachrow(rawdata[2:end, :])
        push!(edges, (Int(d1), Int(d2)).+1)
    end
    (nv=nv, edges=edges)
end

"""Load edge-weighted graph data."""
function loadewg(datapath)
    rawdata = readdlm(datapath)
    nv = Int(rawdata[1, 1])
    ne = Int(rawdata[2, 1])
    edges = Tuple{Int, Int, Float64}[]
    for (d1, d2, d3) in eachrow(rawdata[3:end, :])
        push!(edges, (Int(d1)+1, Int(d2)+1, d3))
    end
    (nv=nv, ne=ne, edges=edges)
end