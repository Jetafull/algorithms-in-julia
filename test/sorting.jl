using Random

include("../src/sorting.jl")

function testsort(sortfunc, testpairs)
    for (arr, ans) in testpairs
        temparr = copy(arr)
        sortfunc(temparr)
        @test temparr == ans 
    end
end

function testrandom(sortfunc, seednum)
        Random.seed!(seednum)
        testdata = randn(100)

        d1 = copy(testdata)
        sort!(d1)
        d2 = copy(testdata)
        sortfunc(d2)

        @test d1 == d2
end


@testset "Sort" begin

    @testset "Basic cases" begin
        testpairs = [
                ([], []),
                ([3, 1, 2], [1, 2, 3]),
                ([4, 2, 3, 1], [1, 2, 3, 4]),
                ([4, 3, 3, 1, -1], [-1, 1, 3, 3, 4]),
                ([1.28, -2.591, -2.2266], [-2.591, -2.2266, 1.28]),
                ([1], [1]),
                (['b', 'c', 'a'], ['a', 'b', 'c']),
                (['E', 'A', 'S', 'Y', 'Q', 'U', 
                  'E', 'S', 'T', 'I', 'O', 'N'],
                ['A', 'E', 'E', 'I', 'N', 'O', 
                 'Q', 'S', 'S', 'T', 'U', 'Y'])
            ]

        testsort(insertsort!, testpairs)
        testsort(mergesort!, testpairs)
        testsort(quicksort!, testpairs)
    end

    @testset "Compare with Base.sort" begin
        for seed = [20190128, 20200828, 20211228]
            testrandom(insertsort!, seed)
            testrandom(mergesort!, seed)
            testrandom(quicksort!, seed)
        end
    end

end




