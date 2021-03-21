include("../src/decisiontree.jl")

@testset "Squared loss function" begin
    y1 = [1 2 3]
    y2 = [3 5 7]

    @test squaredloss(y1) == 2
    @test squaredloss(y2) == 8
    @test squaredloss(y1, y2) == 10
end

@testset "Gini index function" begin
    y1 = [0 0 0 1]
    y2 = [0 0 0 1 1 1 1 1]

    @test giniindex(y1) == 0.375 
    @test giniindex(y2) == 0.46875 
    @test giniindex(y1, y2) == 0.4375 
end

@testset "Classification tree" begin
    @testset "Test tree stopping rule" begin
        X = [[1.0 1.0]; [2.0 1.5]; [3.0 1.7]]
        y = [0, 0, 1]

        clstree = ClassificationTree(X, y, giniindex, TreeStoppingRule(3, 2))
        predictions = predict(clstree, X)
        @test clstree.numregions == 1

        clstree = ClassificationTree(X, y, giniindex, TreeStoppingRule(3, 1))
        predictions = predict(clstree, X)
        @test clstree.numregions == 2
    end

    @testset "Baseline case" begin
        X = [[1.0 1.0]; [2.0 1.5]; [3.0 1.7]; [4.0 2.0];[5.0 3.5]; [6.0 4.0]; [7.0 4.5]]
        y = [0, 0, 1, 1, 1, 0, 0]

        clstree = ClassificationTree(X, y, giniindex, TreeStoppingRule(3, 1))
        predictions = predict(clstree, X)
        
        @test typeof(predictions) == Array{Int64, 1}
        @test predictions == [0, 0, 1, 1, 1, 0, 0]
        @test clstree.numregions == 3
    end
end

@testset "RegressionTree" begin

    @testset "Baseline case" begin
        X = [[1.0 1.0]; [2.0 1.5]; [3.0 1.7]; [4.0 2.0];[5.0 3.5]; [6.0 4.0]; [7.0 4.5]]
        y = [1, 2, 3, 4, 5, 6, 7]

        @testset "with squared loss" begin
            regtree = RegressionTree(X, y, TreeStoppingRule(3, 1))
            predictions = predict(regtree, X)

            @test typeof(predictions) == Array{Float64, 1}
            @test predictions == [1.0, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5]
            @test regtree.numregions == 4
        end
    end

end
