include("../src/searching.jl")
using .Searching

function testbinarysearch(testcase, searchfunc)
    v, k, ans = testcase
    @test searchfunc(k, v) == ans
end

@testset "Binary search" begin
    testcases = [
        ([1, 2, 5], 5, 3),
        ([1, 2, 5], 4, -1), 
        ([1], 1, 1), 
        ([1], 5, -1), 
        ([-1, 0, 3, 5, 10, 200], -5, -1),
        ([-1, 0, 3, 5, 10, 200], 10, 5),
    ]

    for testcase in testcases
        testbinarysearch(testcase, binarysearch)
        testbinarysearch(testcase, 
            (k, v, algo=:iterative) -> binarysearch(k, v, algo=algo))
    end
end


function testactions(testcase, bst)
    actions, ans = testcase
    node = bst.root 

    for action in actions
        if action == :left
            node = node.left
        elseif action == :right
            node = node.right
        end
    end

    if isnothing(ans)
        @test isnothing(node)
    else
        @test node.val == ans
    end
end

@testset "Binary search tree" begin
    v = [5, 3, 6, 7, 4, 10]

    testcases = [
        (actions=[],  ans=5), 
        (actions=[:left, :left],  ans=nothing),
        (actions=[:left, :right], ans=4),
        (actions=[:right, :right, :right], ans=10)
    ]

    @testset "Basics" begin
        bst = BinarySearchTree(v)
        @test Searching.size(bst) == 6

        @test Searching.get(bst, 5).val == 5
        @test isnothing(Searching.get(bst, 100))

        @test Searching.maximum(bst).val == 10
        @test Searching.minimum(bst).val == 3

        for testcase in testcases
            testactions(testcase, bst)
        end
    end

    @testset "Test put!" begin
        bst = BinarySearchTree(v)
        result = Searching.put!(bst, 2)
        @test Searching.size(result) == 7
        @test result.root.left.left.val == 2
    end

    @testset "Test delete!" begin
        bst = BinarySearchTree(v)
        result = Searching.delete!(bst, 5)
        @test result.root.val == 6
        @test result.root.numnodes == 5

        bst = BinarySearchTree(v)
        result = Searching.delete!(bst, 6)
        @test result.root.val == 5 
        @test result.root.numnodes == 5
        @test result.root.right.val == 7

        bst = BinarySearchTree(v)
        bst = Searching.delete!(bst, 5)
        bst = Searching.delete!(bst, 6)
        @test bst.root.val == 7
        @test Searching.size(bst.root) == 4 
        @test bst.root.right.val == 10
    end

    @testset "Test delete! side effects" begin
        # suppose to change the node the same way as returned value
        node = BinarySearchTree(v).root
        Searching.delete!(node, 5)
        result = BinarySearchTree(node)
        @test Searching.size(result) == 5
        @test result.root.val == 6

        node = BinarySearchTree(v).root
        Searching.delete!(node, 6)
        result = BinarySearchTree(node)
        @test Searching.size(result) == 5
        @test result.root.val == 5 
        @test result.root.right.val == 7 
    end

    @testset "Test rank and select" begin
        bst = BinarySearchTree(v)
        @test Searching.rank(bst, 5) == 3
        @test Searching.rank(bst, 1) == 1
        @test Searching.rank(bst, 100) == 7 

        bst = BinarySearchTree(v)
        @test Searching.select(bst, 5).val == 7
        @test Searching.select(bst, 1).val == 3
        @test isnothing(Searching.select(bst, 0))
        @test isnothing(Searching.select(bst, 100))
    end
end
    