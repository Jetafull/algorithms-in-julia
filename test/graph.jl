include("../src/graph.jl")

@testset "Graph search algorithms" begin
    # vertice "7" and "8" is not connected
    examplegraph = (
        nv=8, 
        edges=[
            (1, 6),
            (3, 5),
            (3, 4),
            (2, 3),
            (1, 2),
            (4, 5),
            (4, 6),
            (1, 3),
            (7, 8)
        ]
    )

    g = GraphAlgo.UndirectedGraph(examplegraph.nv, examplegraph.edges)

    @testset "Test depth first search" begin
        # source vertex is "1"
        result = GraphAlgo.search(g, 1, GraphAlgo.FindingPaths, GraphAlgo.DepthFirstSearch)
        @test length(result.marked) == g.numvertices 
        @test sum(result.marked) == 6 
        @test result.numvisited == 6 
        @test result.edgeto == [nothing, 3, 4, 6, 3, 1, nothing, nothing]
        @test GraphAlgo.haspathto(result, 5) == true
        @test GraphAlgo.haspathto(result, 7) == false
        @test GraphAlgo.pathto(result, 3) == [1, 6, 4, 3]
    end

    @testset "Test breadth first search" begin
        result = GraphAlgo.search(g, 1, GraphAlgo.FindingPaths, GraphAlgo.BreadthFirstSearch)
        @test length(result.marked) == g.numvertices 
        @test sum(result.marked) == 6 
        @test result.numvisited == 6 
        @test result.edgeto == [nothing, 1, 1, 6, 3, 1, nothing, nothing]
        @test GraphAlgo.haspathto(result, 2) == true
        @test GraphAlgo.haspathto(result, 7) == false
        @test GraphAlgo.pathto(result, 3) == [1, 3]
        @test GraphAlgo.pathto(result, 4) == [1, 6, 4]
        @test GraphAlgo.pathto(result, 5) == [1, 3, 5]
    end

    @testset "Connected components" begin
        cc = GraphAlgo.search(g, GraphAlgo.ConnectedComponents)
        @test sum(cc.marked) == g.numvertices
        @test cc.vids[6] == 1
        @test cc.vids[7] == 2
    end
    
    @testset "Cycle detection" begin
        @test GraphAlgo.search(g, GraphAlgo.Cycle).hascycle == true

        g = GraphAlgo.UndirectedGraph(3, [(1,2), (2,3)])
        @test GraphAlgo.search(g, GraphAlgo.Cycle).hascycle == false

        examplegraph = (
        nv=8, 
        edges=[
            (1, 6),
            (3, 5),
            (1, 2),
            (4, 5),
            (4, 6),
            (7, 8)
            ]
        )

        g = GraphAlgo.UndirectedGraph(examplegraph.nv, examplegraph.edges)
        @test GraphAlgo.search(g, GraphAlgo.Cycle).hascycle == false
    end

end

@testset "Directed Graph" begin
    minidg = (
        nv = 6,
        ne = 4,
        edges = [
            (6, 5),
            (5, 4),
            (4, 6),
            (1, 6),
        ]
    )

    minidg2 = (
        nv = 6,
        ne = 4,
        edges = [
            (6, 5),
            (5, 4),
            (1, 6),
        ]
    )   


    # tinyDG.txt from *Algorithm 4th* page 569
    tinydg = (
        nv = 13,
        ne = 22,
        edges = [
            ( 5,  3),
            ( 3,  4),
            ( 4,  3),
            ( 7,  1),
            ( 1,  2),
            ( 3,  1), 
            (12, 13),
            (13, 10),
            (10, 11),
            (10, 12),
            ( 9, 10),
            (11, 13),
            (12,  5),
            ( 5,  4),
            ( 4,  6),
            ( 8,  9),
            ( 9,  8),
            ( 6,  5),
            ( 1,  6),
            ( 7,  5),
            ( 7, 10),
            ( 8,  7)
        ]
    )

    # DAG examle from p579 in *Algorithm 4th*.
    dag_p579 = (
        nv = 13,
        ne = 15,
        edges = [
            (1, 2),
            (1, 6),
            (1, 7),
            (3, 1),
            (3, 4),
            (4, 6),
            (6, 5),
            (7, 5),
            (7, 10),
            (8, 7),
            (9, 8),
            (10, 11),
            (10, 12),
            (10, 13),
            (12, 13)
        ]
    )

    g = GraphAlgo.DirectedGraph(tinydg.nv, tinydg.edges)

    @test g.numedges == tinydg.ne

    fp_result = GraphAlgo.search(g, 2, GraphAlgo.FindingPaths)
    @test length(fp_result.marked) == g.numvertices
    @test fp_result.numvisited == 1

    fp_result = GraphAlgo.search(g, 3, GraphAlgo.FindingPaths)
    @test fp_result.marked[1:6] == fill(true, 6)
    @test fp_result.marked[7:end] == fill(false, 7)

    fp_result = GraphAlgo.search(g, 3, GraphAlgo.FindingPaths, GraphAlgo.BreadthFirstSearch)
    @test fp_result.marked[1:6] == fill(true, 6)
    @test fp_result.marked[7:end] == fill(false, 7)

    # test whether breadth first search find the shortest path
    fp_result = GraphAlgo.search(g, 7, GraphAlgo.FindingPaths, GraphAlgo.DepthFirstSearch)
    @test (GraphAlgo.pathto(fp_result, 3) == [7, 1, 6, 5, 3] ||
           GraphAlgo.pathto(fp_result, 3) == [7, 5, 3])
    fp_result = GraphAlgo.search(g, 7, GraphAlgo.FindingPaths, GraphAlgo.BreadthFirstSearch)
    @test GraphAlgo.pathto(fp_result, 3) == [7, 5, 3]

    marked = GraphAlgo.search(g, [2, 3, 7], GraphAlgo.MultiFindingPaths)
    @test marked == Bool[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]

    g = GraphAlgo.DirectedGraph(minidg.nv, minidg.edges)
    cycle = GraphAlgo.search(g, GraphAlgo.Cycle)
    @test cycle.trace == [6, 5, 4, 6]
    @test cycle.hascycle == true

    g = GraphAlgo.DirectedGraph(minidg2.nv, minidg2.edges)
    cycle = GraphAlgo.search(g, GraphAlgo.Cycle)
    @test isnothing(cycle.trace)
    @test cycle.hascycle == false 

    g = GraphAlgo.DirectedGraph(tinydg.nv, tinydg.edges)
    cycle = GraphAlgo.search(g, GraphAlgo.Cycle)
    @test isnothing(cycle.trace) == false
    @test cycle.hascycle == true 

    # Test topological sort
    g = GraphAlgo.DirectedGraph(tinydg)
    result = GraphAlgo.search(g, GraphAlgo.Topological)
    @test isnothing(result.vorder)

    g = GraphAlgo.DirectedGraph(dag_p579)
    result = GraphAlgo.search(g, GraphAlgo.Topological)
    @test result.vorder == [9, 8, 3, 4, 1, 7, 10, 12, 13, 11, 6, 5, 2]

    g = GraphAlgo.DirectedGraph(tinydg)
    result = GraphAlgo.search(g, GraphAlgo.ConnectedComponents)
    @test Set(result.vids) == Set([1, 2, 3, 4, 5])
    @test length(Set(result.vids[10:13])) == 1
    @test length(Set(result.vids[[1, 3, 4, 5, 6]])) == 1
end