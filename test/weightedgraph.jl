using Test

include("../src/weightedgraph.jl")
include("../test/utils.jl")
using .WeightedGraphAlgo
using .WeightedGraphAlgo: WeightedUndirectedEdge

const tiny_ewg_p614 = (
    nv = 8,
    ne = 16,
    edges = [
        (5, 6, 0.35),
        (5, 8, 0.37),
        (6, 8, 0.28),
        (1, 8, 0.16),
        (2, 6, 0.32),
        (1, 5, 0.38),
        (3, 4, 0.17),
        (2, 8, 0.19),
        (1, 3, 0.26),
        (2, 3, 0.36),
        (2, 4, 0.29),
        (3, 8, 0.34),
        (7, 3, 0.40),
        (4, 7, 0.52),
        (7, 1, 0.58),
        (7, 4, 0.93)
    ]
)

@testset "Weighted undirected graph" begin

    @testset "Lazy Prim algorithm" begin
        g = WeightedUndirectedGraph(tiny_ewg_p614)
        mst = MST(g, LazyPrimMST)
        edgeset = Set(mst.result.mstedges)
        @test sum(mst.result.mstvertices) == g.numvertices
        @test sumweights(mst) == 1.81
        @test WeightedUndirectedEdge((1, 8, 0.16)) in edgeset
        @test WeightedUndirectedEdge((6, 8, 0.28)) in edgeset
    end

    @testset "Eager Prim algorithm" begin
        g = WeightedUndirectedGraph(tiny_ewg_p614)
        mst = MST(g, EagerPrimMST)
        edgeset = Set(mst.result.mstedges)
        @test sum(mst.result.mstvertices) == g.numvertices
        @test sumweights(mst) ≈ 1.81
        @test WeightedUndirectedEdge((1, 8, 0.16)) in edgeset
        @test WeightedUndirectedEdge((6, 8, 0.28)) in edgeset
    end

    @testset "Kruskal's algorithm" begin
        g = WeightedUndirectedGraph(tiny_ewg_p614)
        mst = MST(g, KruskalMST)
        edgeset = Set(mst.result.mstedges)
        @test sumweights(mst) == 1.81
        @test WeightedUndirectedEdge((1, 8, 0.16)) in edgeset
        @test WeightedUndirectedEdge((6, 8, 0.28)) in edgeset
    end

end

@testset "Weighted directed graph" begin
    CURRENT_PATH = cd(pwd, ".")
    tinyewd = loadewg(joinpath(CURRENT_PATH, "data/tinyEWD.txt"))

    @testset "Dijkstra shortest path" begin
        g = WeightedDirectedGraph(tinyewd)
        sp = SP(g, 1, DijkstraSP)
        @test distanceto(sp, 2) ≈ 1.05
        @test distanceto(sp, 3) ≈ 0.26 
        @test distanceto(sp, 4) ≈ 0.99 
        @test distanceto(sp, 5) ≈ 0.38 
        @test distanceto(sp, 6) ≈ 0.73 
        @test distanceto(sp, 7) ≈ 1.51 
        @test distanceto(sp, 8) ≈ 0.60 
    end

end