include("./utils.jl")
include("../src/unionfind.jl")
using .UnionFindAlgo

PROJECT_PATH = joinpath(@__DIR__, "..")
tinyuf = loadufdata(joinpath(PROJECT_PATH, "data/tinyUF.txt"))
mediumuf = loadufdata(joinpath(PROJECT_PATH, "data/mediumUF.txt"))


@testset "Union find" begin
    uf = UnionFind(tinyuf)
    @test uf.ncomp == 2
    @test isconnected(uf, 3, 4) == false
    @test isconnected(uf, 4, 5) == true
    @test isconnected(uf, 9, 10) == true

    uf = UnionFind(mediumuf)
    @test uf.ncomp == 3
end