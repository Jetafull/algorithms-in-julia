include("../src/heapqueue.jl")
using .HeapQueueModule

testdata = [30, 10, 35, 15, 12, 40, 23, 20]

@testset "min heap queue works" begin
    minhq = heapify(testdata, Min)

    @test popfirst!(minhq) == 10
    @test popfirst!(minhq) == 12 
    @test popfirst!(minhq) == 15 

    minhq = heapify(testdata, Min)
    for _ = 1:(length(minhq)-1)
        popfirst!(minhq)
    end
    @test popfirst!(minhq) == 40 
end

@testset "max heap queue works" begin
    maxhq = heapify(testdata, Max)
    @test popfirst!(maxhq) == 40 
    @test popfirst!(maxhq) == 35 
    @test popfirst!(maxhq) == 30 

    maxhq = heapify(testdata, Max)
    for _ = 1:(length(maxhq)-1)
        popfirst!(maxhq)
    end
    @test popfirst!(maxhq) == 10
end

@testset "test heapify!" begin
    data = deepcopy(testdata)
    hq = heapify!(data, Min)
    result = Int[]
    while !isempty(data)
        push!(result, popfirst!(hq))
    end
    @test result == [10, 12, 15, 20, 23, 30, 35, 40]
end

function testsort(sortfunc, testpairs)
    for (arr, ans) in testpairs
        temparr = deepcopy(arr)
        sortfunc(temparr)
        @test temparr == ans 
    end
end

@testset "heapsort" begin
    testpairs = [
            ([], []),
            ([1], [1]),
            ([3, 1, 2], [1, 2, 3]),
            ([1, 3, 2], [1, 2, 3]),
            ([1, 2, 3], [1, 2, 3]),
            ([4, 2, 3, 1], [1, 2, 3, 4]),
            ([4, 3, 3, 1, -1], [-1, 1, 3, 3, 4]),
            (['b', 'c', 'a'], ['a', 'b', 'c']),
            (['E', 'A', 'S', 'Y', 'Q', 'U', 
              'E', 'S', 'T', 'I', 'O', 'N'],
             ['A', 'E', 'E', 'I', 'N', 'O', 
              'Q', 'S', 'S', 'T', 'U', 'Y'])
        ]

    testsort(heapsort!, testpairs)
end

@testset "Indexed heap sort" begin
    m1 = Base.split("A B C F G I I Z")
    m2 = Base.split("B D H P Q Q")
    m3 = Base.split("A B E F J N")
    orig_streams = [m1, m2, m3]

    @testset "Indexed Min HQ" begin
        # the multiway merge example on: 
        # https://algs4.cs.princeton.edu/24pq/Multiway.java.html
        output = []
        streams = deepcopy(orig_streams)
        hq = HeapQueueModule.IndexedMinHQ{String}(3)
        HeapQueueModule.insert!(hq, 1, popfirst!(streams[1])) 
        while !isempty(hq)
            for i in 1:3
                if !HeapQueueModule.contains(hq, i) && !isempty(streams[i])
                    new_elem = popfirst!(streams[i])
                    HeapQueueModule.insert!(hq, i, new_elem)
                end
            end
            _, firstkey = HeapQueueModule.popfirst!(hq)
            push!(output, firstkey)
        end

        @test join(output, " ") == "A A B B B C D E F F G H I I J N P Q Q Z"
    end

    @testset "Test change key" begin
        hq = HeapQueueModule.IndexedMinHQ{Float64}(5)
        HeapQueueModule.insert!(hq, 1, 3)
        HeapQueueModule.insert!(hq, 2, 1)
        HeapQueueModule.insert!(hq, 3, 2)
        @test_throws HeapQueueModule.InvalidKeyChangeError decreasekey!(hq, 3, 3)
        
        decreasekey!(hq, 3, 0.5)
        @test HeapQueueModule.popfirst!(hq) == (3, 0.5)
    end




end