module WeightedGraphAlgo

export 
    WeightedUndirectedGraph,
    WeightedDirectedGraph,
    MST,
    sumweights,
    LazyPrimMST,
    EagerPrimMST,
    KruskalMST,
    SP,
    DijkstraSP,
    distanceto

include("./heapqueue.jl")
using .HeapQueueModule: MinHeapQueue, heapify!, IndexedMinHQ, decreasekey!, 
    contains, insert!, isempty, popfirst!

include("./unionfind.jl")
using .UnionFindAlgo: UnionFind, isconnected, union!

abstract type Edge end
abstract type WeightedEdge <: Edge end

EdgeData = Tuple{Int, Int, Float64}

# Weighted Graph Data Structure
struct WeightedUndirectedEdge <: WeightedEdge
    v1::Int
    v2::Int
    weight::Float64
    WeightedUndirectedEdge(edgedata::EdgeData) = new(edgedata...)
end

struct WeightedDirectedEdge <: WeightedEdge
    v1::Int
    v2::Int
    weight::Float64
    WeightedDirectedEdge(edgedata::EdgeData) = new(edgedata...)
end

Base.:>(e1::T, e2::T) where T<:WeightedEdge = e1.weight > e2.weight
Base.:<(e1::T, e2::T) where T<:WeightedEdge = e1.weight < e2.weight
Base.:(==)(e1::T, e2::T) where T<:WeightedEdge = e1.weight == e2.weight
Base.show(io::IO, edge::WeightedEdge) = print(io, split(string(typeof(edge)), ".")[end], 
                                              "($(edge.v1), $(edge.v2), $(edge.weight))")

mutable struct Graph{T<:WeightedEdge}
    numvertices::Int
    numedges::Int
    adjlist::Vector{Vector{T}}
    function Graph{T}(numvertices::Int) where T 
        new(numvertices, 0, [T[] for _ = 1:numvertices])
    end
end
Base.show(io::IO, g::Graph{T}) where T = print(io, 
                    "Graph{$(split(string(T), ".")[end])}",
                    "(V=$(g.numvertices), E=$(g.numedges))") 

function WeightedUndirectedGraph(graphdata::NamedTuple)
    g = Graph{WeightedUndirectedEdge}(graphdata.nv)
    addedges!(g, graphdata.edges)
    g
end

function WeightedDirectedGraph(graphdata::NamedTuple)
    g = Graph{WeightedDirectedEdge}(graphdata.nv)
    addedges!(g, graphdata.edges)
    g
  end

function addedge!(g::Graph{T}, edge::T) where T<:WeightedUndirectedEdge 
    push!(g.adjlist[edge.v1], edge)
    push!(g.adjlist[edge.v2], edge)
    g.numedges += 1
end

function addedge!(g::Graph{T}, edge::T) where T<:WeightedDirectedEdge 
    push!(g.adjlist[edge.v1], edge)
    g.numedges += 1
end

function addedges!(g::Graph{T}, edgesdata::Vector{EdgeData}) where {T<:WeightedEdge}
    for edgedata in edgesdata 
        edge = T(edgedata)
        addedge!(g, edge)
    end
end

"""
    getedges(g)

Get edges from the adjacent edge list.
"""
function getedges(g::Graph{T}) where T<:WeightedEdge
    edgelist = T[] 
    for (v, adjedges) in enumerate(g.adjlist)
        for edge in adjedges
            pushedge!(edgelist, edge, v)
        end
    end
    edgelist
end

function pushedge!(edgelist::Vector{T}, edge::T, v) where T<:WeightedUndirectedEdge  
    findadjv(edge, v) < v && push!(edgelist, edge) 
end
pushedge!(edgelist::Vector{T}, edge::T, v) where T<:WeightedDirectedEdge = 
    push!(edgelist, edge)

struct UnfoundVertex <: Exception 
    v
    edge
end
Base.showerror(io::IO, e::UnfoundVertex) = print(io, "vertex $(e.v) not found on $(e.edge)")
function findadjv(edge::WeightedUndirectedEdge, v) 
    v == edge.v1 && return edge.v2
    v == edge.v2 && return edge.v1
    throw(UnfoundVertex(v, edge))
end

"""
    findadje(g, v) 

Find adjacenwt edges of vertex `v` in graph `g`.
"""
findadje(g::Graph, v) =  g.adjlist[v]

# Minimum Spanning Tree (MST) problem
abstract type MSTResult end

abstract type MSTAlgo end
abstract type PrimMSTAlgo <: MSTAlgo end
struct LazyPrimMSTAlgo <: PrimMSTAlgo end
struct EagerPrimMSTAlgo <: PrimMSTAlgo end
struct KruskalMSTAlgo <: MSTAlgo end

const LazyPrimMST = LazyPrimMSTAlgo()
const EagerPrimMST = EagerPrimMSTAlgo()
const KruskalMST = KruskalMSTAlgo()
struct MST
    result::MSTResult
    algo::MSTAlgo

    function MST(g, algo)
        result = createresult(g, algo)
        findmst!(g, result, algo)
        new(result, algo)
    end
end

sumweights(mst) = sum([e.weight for e in mst.result.mstedges if !isnothing(e)])

struct LazyPrimMSTResult <: MSTResult
    mstvertices::Vector{Bool}
    mstedges::Vector{WeightedUndirectedEdge}
    crossingedges::MinHeapQueue{WeightedUndirectedEdge}

    function LazyPrimMSTResult(g)
        mstvertices = [false for _ in 1:g.numvertices]
        mstedges = WeightedUndirectedEdge[]
        crossingedges = MinHeapQueue{WeightedUndirectedEdge}()
        new(mstvertices, mstedges, crossingedges)
    end
end

"""
Eager Prim algorithm result.

- `mstedges`: `mstedges[v]` is the edge connecting v to the MST.
- `distto`: `distto[v]` is the weight of `mstedges[v]`
- `vqueue`: an indexed minimum heap queue of vertex with weight as prority.
"""
struct EagerPrimMSTResult <: MSTResult
    mstvertices::Vector{Bool}
    mstedges::Vector{Union{WeightedUndirectedEdge, Nothing}}
    distto::Vector{Float64}
    vqueue::IndexedMinHQ{Float64}

    function EagerPrimMSTResult(g)
        mstvertices = [false for _ in 1:g.numvertices]
        mstedges = Vector{Union{WeightedUndirectedEdge, Nothing}}(nothing, g.numvertices)
        distto = [Inf for _ in 1:g.numvertices]
        vqueue = IndexedMinHQ{Float64}(g.numvertices)
        new(mstvertices, mstedges, distto, vqueue)
    end
end

struct KruskalMSTResult <: MSTResult
    mstvertices::UnionFind
    mstedges::Vector{WeightedUndirectedEdge}
    edgequeue::MinHeapQueue{WeightedUndirectedEdge}

    function KruskalMSTResult(g)
        mstvertices = UnionFind(g.numvertices) 
        mstedges = WeightedUndirectedEdge[]
        # create a min heap queue from edge list
        edgequeue = heapify!(getedges(g)) 
        new(mstvertices, mstedges, edgequeue)
    end
end
    
createresult(g, ::LazyPrimMSTAlgo) = LazyPrimMSTResult(g)
createresult(g, ::EagerPrimMSTAlgo) = EagerPrimMSTResult(g)
createresult(g, ::KruskalMSTAlgo) = KruskalMSTResult(g)

# Lazy Prim algorithm
function findmst!(g, result, algo::LazyPrimMSTAlgo)
    visit!(g, 1, result)
    while !isempty(result.crossingedges)
        e = popfirst!(result.crossingedges)
        iseligible(result.mstvertices, e) || continue 
        push!(result.mstedges, e)
        ismarked(result.mstvertices, e.v1) || visit!(g, e.v1, result)
        ismarked(result.mstvertices, e.v2) || visit!(g, e.v2, result)
    end
end

function visit!(g, v, result::LazyPrimMSTResult)
    result.mstvertices[v] = true
    for e in findadje(g, v)
        adjv = findadjv(e, v)
        # find the adjacent vertex on the same edge
        # if the adjacent vertex is not on the tree,
        # then this edge is crossing edge
        ismarked(result.mstvertices, adjv) || push!(result.crossingedges, e)
    end
end

iseligible(mstvertices, edge::Edge) = mstvertices[edge.v1] == false || mstvertices[edge.v2] == false
ismarked(mstvertices, v) = mstvertices[v] == true

# Eager Prim Algorithm
function findmst!(g, result, ::EagerPrimMSTAlgo)
    result.distto[1] = 0.0
    insert!(result.vqueue, 1, 0.0)
    while !isempty(result.vqueue)
        minv, _ = popfirst!(result.vqueue)
        visit!(g, minv, result)
    end
end

function visit!(g, v, result::EagerPrimMSTResult)
    result.mstvertices[v] = true
    for e in findadje(g, v)
        adjv = findadjv(e, v)
        # adjv is already in the MST, ineligible edge
        ismarked(result.mstvertices, adjv) && continue
        if e.weight < result.distto[adjv] 
            result.mstedges[adjv] = e
            result.distto[adjv] = e.weight
            if contains(result.vqueue, adjv)
                # if adjv exists in pq, make sure the edge weight 
                # is smaller than existing value
                decreasekey!(result.vqueue, adjv, result.distto[adjv])
            else
                insert!(result.vqueue, adjv, result.distto[adjv])
            end
        end
    end
end

# Kruskal's algorithm
function findmst!(g, result, algo::KruskalMSTAlgo)
    while !isempty(result.edgequeue) && length(result.mstedges) < g.numvertices - 1
        edge = popfirst!(result.edgequeue)
        # if v1 and v2 are not connected in the subtree of MST
        # that means the edge is a non-black crossing edge with minimum 
        # weight (because it's from minimum priority queue) 
        # according to greedy MST algorithm. 
        # this edge must be on MST of the graph
        if !isconnected(result.mstvertices, edge.v1, edge.v2)
            push!(result.mstedges, edge)
            union!(result.mstvertices, edge.v1, edge.v2)
        end
    end
end

# Shortest Paths
abstract type ShortestPathsAlgo end
abstract type ShortestPathsResult end
struct DijkstraSPAlgo <: ShortestPathsAlgo end
const DijkstraSP = DijkstraSPAlgo()


struct SP
    result::ShortestPathsResult
    algo::ShortestPathsAlgo
    function SP(g, source, algo::ShortestPathsAlgo)
        validate(g, algo)
        result = createresult(g, source, algo)
        findsp!(g, source, result, algo)
        new(result, algo)
    end
end

function validate(g::Graph{WeightedDirectedEdge}, ::DijkstraSPAlgo)
    for (v, adjedges) in enumerate(g.adjlist)
        for edge in adjedges
            edge.weight < 0 && throw(ArgumentError(
                "$(edge.weight): Dijkastra shortst paths requires non-negative weight"))
        end
    end
end
createresult(g::Graph{WeightedDirectedEdge}, source, ::DijkstraSPAlgo) = DijkstraSPResult(g, source) 
    
mutable struct DijkstraSPResult <: ShortestPathsResult
    marked::Vector{Bool}
    edgeto::Vector{Union{Int,Nothing}}
    distto::Vector{Union{Float64,Nothing}}
    vqueue::IndexedMinHQ{Float64}
    source::Int

    function DijkstraSPResult(g::Graph{T}, source) where T<:WeightedDirectedEdge
        marked = [false for _ in 1:g.numvertices]
        mstedges = Vector{Union{T, Nothing}}(nothing, g.numvertices)
        distto = [Inf for _ in 1:g.numvertices]
        vqueue = IndexedMinHQ{Float64}(g.numvertices)
        new(marked, mstedges, distto, vqueue, source)
    end
end

function findsp!(g::Graph{WeightedDirectedEdge}, source, result, ::DijkstraSPAlgo)
    disttosource = 0.0
    result.distto[source] = disttosource 
    insert!(result.vqueue, source, disttosource)
    while !isempty(result.vqueue)
        minv, _ = popfirst!(result.vqueue)
        relax!(g, minv, result)
    end
end

function relax!(g::Graph{WeightedDirectedEdge}, v, result::DijkstraSPResult)
    result.marked[v] = true 
    for e in findadje(g, v)
        tov = findtov(e, v)
        result.marked[tov] && continue
        newdist = result.distto[v] + e.weight
        if newdist < result.distto[tov]
            result.edgeto[tov] = v
            result.distto[tov] = newdist 
            if !contains(result.vqueue, tov)
                insert!(result.vqueue, tov, newdist)
            else
                decreasekey!(result.vqueue, tov, newdist)
            end
        end
    end
end

function findtov(e::WeightedDirectedEdge, v) 
    e.v1 == v || throw(InvalidArgumentError("$v is not the from vertice"))
    e.v2
end

distanceto(sp::SP, tov) = sp.result.distto[tov]

end # module