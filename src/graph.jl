module GraphAlgo

Edge = NTuple{2, Int}
abstract type GraphType end
struct Undirected <: GraphType end
struct Directed <: GraphType end

mutable struct Graph{T}
    numvertices::Int
    numedges::Int
    adjlist::Vector{Vector{Int}}

    function Graph{T}(numvertices::Int) where T <: GraphType
        # don't use `fill`, which will fill with the same array reference
        adjlist = [Vector{Int}() for i = 1:numvertices]
        new{T}(numvertices, 0, adjlist)
    end
end

function Graph{T}(numvertices::Int, edges::Vector{Edge}) where T
    g = Graph{T}(numvertices)
    for edge in edges
        addedge!(g, edge)
    end
    g
end

function Graph{T}(graphdata::NamedTuple) where T <: GraphType
    Graph{T}(graphdata.nv, graphdata.edges)
end

function addedge!(g::Graph{Undirected}, edge::Edge)
    push!(g.adjlist[edge[1]], edge[2])
    push!(g.adjlist[edge[2]], edge[1])
    g.numedges += 1
    nothing
end

function addedge!(g::Graph{Directed}, edge::Edge)
    push!(g.adjlist[edge[1]], edge[2])
    g.numedges += 1
    nothing
end

function reverse(g::Graph{Directed})
    r = Graph{Directed}(g.numvertices)
    for v in 1:g.numvertices
        for adjv in findadj(g, v)
            # add edge in reversed order
            addedge!(r, (adjv, v))
        end
    end
    return r 
end

UndirectedGraph(numvertices::Int, edge::Vector{Edge}) = Graph{Undirected}(numvertices, edge)
UndirectedGraph(graphdata::NamedTuple) = Graph{Undirected}(graphdata)
DirectedGraph(numvertices::Int, edge::Vector{Edge}) = Graph{Directed}(numvertices, edge)
DirectedGraph(graphdata::NamedTuple) = Graph{Directed}(graphdata)

"""
    findadj(g, v) 

Find adjacent vertices of vertex `v` in graph `g`.
"""
findadj(g, v) =  g.adjlist[v]

abstract type GraphSearchAlgo end
struct DepthFirstSearchAlgo <: GraphSearchAlgo end
struct BreadthFirstSearchAlgo <: GraphSearchAlgo end

const DepthFirstSearch = DepthFirstSearchAlgo()
const BreadthFirstSearch = BreadthFirstSearchAlgo()

# With the graph data structure, we can answer several queries:
# 1. `FindingPaths`: given a source vertex, find paths to all the other vertices.
# 2. `ConnectedComponents`: Find the number of connected components, and label each
#    vertex to one of the components.
abstract type GraphQuery end
abstract type GraphQueryResult end

# Finding path example in Algorithm 4th
# Single-source finding paths
struct FindingPathsQuery <: GraphQuery end
const FindingPaths = FindingPathsQuery()
# Multi-source finding paths
struct MultiFindingPathsQuery <: GraphQuery end
const MultiFindingPaths = MultiFindingPathsQuery()

mutable struct FindingPathsResult <: GraphQueryResult 
    source::Int
    marked::Vector{Bool}
    numvisited::Int
    edgeto::Vector{Union{Int, Nothing}} # v-w: where w is the last vertice coming to v

    function FindingPathsResult(g, s)
        nv = g.numvertices
        new(s, Bool[0 for _ = 1:nv], 0, Vector{Union{Int, Nothing}}(nothing, nv))
    end
end

function search!(g, v, fp::FindingPathsResult, algo::DepthFirstSearchAlgo)
    mark!(fp, v)
    for adjv = findadj(g, v)
        if !fp.marked[adjv]
            search!(g, adjv, fp, algo)
            fp.edgeto[adjv] = v
        end
    end
end

function mark!(fp::FindingPathsResult, v)
    fp.marked[v] = true
    fp.numvisited += 1
    nothing
end

"""
Breadth first search searches the current vertex and its adjacent vertices.
The adjacent vertices are pushed into a queue and checked in sequence.
"""
function search!(g, v, fp::FindingPathsResult, ::BreadthFirstSearchAlgo)
    searchqueue = [v]  
    mark!(fp, v)

    while !isempty(searchqueue)
        v = popfirst!(searchqueue)
        for adjv in findadj(g, v)
            if !fp.marked[adjv]
                mark!(fp, adjv)
                fp.edgeto[adjv] = v
                push!(searchqueue, adjv)
            end
        end
    end

    nothing
end

function search(g, s::Int, ::FindingPathsQuery, algo::GraphSearchAlgo=DepthFirstSearch)
    fp = FindingPathsResult(g, s)
    search!(g, s, fp, algo)
    fp
end

"""
    multifindpaths(g, slist; searchalgo)

Find paths for multiple sources. The first returned value is the reachablility 
from all sources in `slist`.
"""
function search(g, slist::Vector{Int}, ::MultiFindingPathsQuery)
    marked = Bool[false for _ in 1:g.numvertices]
    for s in slist
        fp = search(g, s, FindingPaths, DepthFirstSearch)
        # element-wise OR 
        marked = marked .| fp.marked
    end
    marked
end

haspathto(fp, v) = fp.marked[v]

function pathto(fp::FindingPathsResult, v)
    if !haspathto(fp, v)
        return nothing
    end
    path = Int[]
    lastv = v
    while !isnothing(lastv)
        pushfirst!(path, lastv)
        lastv = fp.edgeto[lastv]
    end

    return path 
end

# Connected component example
struct ConnectedComponentsQuery <: GraphQuery end
const ConnectedComponents = ConnectedComponentsQuery()
mutable struct ConnectedComponentsResult <: GraphQueryResult
    vids::Vector{Union{Int, Missing}}
    marked::Vector{Bool}
    compcount::Int

    function ConnectedComponentsResult(g)
        new(Vector{Union{Int, Missing}}(missing, g.numvertices), 
            Bool[0 for _ in 1:g.numvertices], 1)
    end
end

function search!(g, v, cc::ConnectedComponentsResult, algo::DepthFirstSearchAlgo)
    cc.marked[v] = true
    cc.vids[v] = cc.compcount

    for adjv in findadj(g, v)
        if !cc.marked[adjv]
            search!(g, adjv, cc, algo)
        end
    end

    nothing
end

function search(g, ::ConnectedComponentsQuery, order::Vector{Int})
    cc = ConnectedComponentsResult(g)
    for v in order 
        if !cc.marked[v]
            search!(g, v, cc, DepthFirstSearch)
            cc.compcount += 1
        end
    end
    cc
end

search(g::Graph{Undirected}, cc::ConnectedComponentsQuery) = search(g, cc, collect(1:g.numvertices))

# Query: Cycle detection
struct CycleQuery <: GraphQuery end
const Cycle = CycleQuery()

mutable struct CycleResult <: GraphQueryResult
    marked::Vector{Bool}
    hascycle::Bool

    CycleResult(g) = new(
        [false for _ in 1:g.numvertices], 
        false)
end
mutable struct DirectedCycleResult <: GraphQueryResult
    marked::Vector{Bool}
    onstack::Vector{Bool}
    edgeto::Vector{Union{Int, Nothing}} # v-w: where w is the last vertice coming to v
    trace::Union{Vector{Int}, Nothing}
    hascycle::Bool

    DirectedCycleResult(g) = new(
        [false for _ in 1:g.numvertices], 
        [false for _ in 1:g.numvertices], 
        Vector{Union{Int, Nothing}}(nothing, g.numvertices),
        nothing,
        false)
end

cycleresult(g::Graph{Directed}) = DirectedCycleResult(g)
cycleresult(g::Graph{Undirected}) = CycleResult(g)

function search(g, ::CycleQuery)
    cycle = cycleresult(g)
    for v = 1:g.numvertices
        cycle.marked[v] || search!(g, v, cycle, DepthFirstSearch)
    end
    cycle
end

"""
    depthfirstsearch(g, v, u, cycle)

Detect cycle in undirected graph with depth first search.
"""
function search!(g::Graph{Undirected}, v, u, cycle::CycleResult, algo::DepthFirstSearchAlgo)
    cycle.marked[v] = true

    for adjv in findadj(g, v)
        if !cycle.marked[adjv]
            search!(g, adjv, v, cycle, algo)
        # the elseif condition means adjv has already been visited
        # if adjv == u, then search goes back to the last 
        # vertex: u -> v -> u
        # if adjv ≠ u, then search goes back to the vertex 
        # before last vertex: adjv -> ... -> u -> v -> adjv  
        # this means there is a cycle starting and ending with `adjv`
        elseif adjv != u
            cycle.hascycle = true
        end
    end

end

function search!(g::Graph{Undirected}, v, cycle::CycleResult, algo::DepthFirstSearchAlgo)
    search!(g, v, v, cycle, algo)
end

function search!(g::Graph{Directed}, v, cycle::DirectedCycleResult, algo::DepthFirstSearchAlgo)
    cycle.marked[v] = true
    # register v on stack
    cycle.onstack[v] = true

    for adjv in findadj(g, v)
        if cycle.hascycle
            return nothing
        elseif !cycle.marked[adjv]
            cycle.edgeto[adjv] = v
            search!(g, adjv, cycle, algo)
        # we come back to a vertex "v*" (adjv here) who is already on the calling stack
        elseif cycle.onstack[adjv]
            cycle.hascycle = true
            # we can trace back from "v*" to "v*" to find the cycle!
            trace = [adjv]
            x = v
            while x != adjv
                # move one step back
                pushfirst!(trace, x)
                x = cycle.edgeto[x]
            end
            pushfirst!(trace, adjv)
            cycle.trace = trace
        end
    end

    # v finishes all recursive calls, unregister from stack
    cycle.onstack[v] = false

    return nothing
end

# Depth first search vertex ordering
# Traverse the graph with depth first search in different ordering.
abstract type GraphSearchOrder end
struct Preorder <: GraphSearchOrder end
struct Postorder <: GraphSearchOrder end
struct ReversePostorder <: GraphSearchOrder end

struct DepthFirstOrderQuery{T<:GraphSearchOrder} <: GraphQuery end

const DepthFirstPreorder = DepthFirstOrderQuery{Preorder}()
const DepthFirstPostorder = DepthFirstOrderQuery{Postorder}()
const DepthFirstReversePostorder = DepthFirstOrderQuery{ReversePostorder}()

struct DepthFirstOrderResult <: GraphQueryResult 
    marked::Vector{Bool}
    vorder::Vector{Int}

    DepthFirstOrderResult(g) = new(Bool[0 for _ in 1:g.numvertices],  Int[])
end

function search(g::Graph{Directed}, query::DepthFirstOrderQuery)
    result = DepthFirstOrderResult(g)
    for v in 1:g.numvertices
        result.marked[v] || search!(g, v, result, query)
    end
    result
end 

function search!(g::Graph{Directed}, v, result::DepthFirstOrderResult, query::DepthFirstOrderQuery{ReversePostorder})
    result.marked[v] = true

    for adjv in findadj(g, v)
        if !result.marked[adjv]
            search!(g, adjv, result, query)
        end
    end
    # reverse postorder
    pushfirst!(result.vorder, v)

    nothing
end

# Topological sort
# See Propsoition F in *Algorithm 4th" page 582:
# Reverse postorder in a DAG is a topological sort.
struct TopologicalQuery <: GraphQuery end
const Topological = TopologicalQuery()

mutable struct TopologicalQueryResult <: GraphQueryResult
    vorder::Union{Vector{Int}, Nothing}
    TopologicalQueryResult() = new(nothing)
end

function search(g::Graph{Directed}, ::TopologicalQuery)
    result = TopologicalQueryResult()
    if search(g, Cycle).hascycle
        return result
    else
        result.vorder = search(g, DepthFirstReversePostorder).vorder
    end
    return result
end

# Strong connectivity for directed graph
"""
    search!(g::Graph{Directed}, ::ConnectedComponents)

Find strongly connected components in a directed graph using 
Kosaruju's algorithm (p584 in *Algorithm 4th*).
"""
function search(g::Graph{Directed}, cc::ConnectedComponentsQuery)
    # Compute the reverse posterorder of g's reverse
    revpostorder = search(reverse(g), DepthFirstReversePostorder).vorder

    # Run standard DFS on G, but consider the unmarked vertices in 
    # the order just computed, instead of standard numerical order
    cc_result = search(g, cc, revpostorder)

    cc_result
end

end # module
