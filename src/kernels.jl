using LinearAlgebra

"""
A kernel function takes two vectors and calculates
the distance between them.
"""
abstract type Kernel end

"""
    SquaredExponentialKernel(l, σf, σy)

Squared exponential kernel for noisy observations.
The parameter names follow (15.20) in MLAPP. 

# Parameters
- `l`: length scale, which controls the horizontal scale. 
      It can be in either of the forms
   1. `Float64`: each dimension will have the same length scale
   2. `Vector{Float64}`: length scale for each dimension.
- `σf`: vertical scale 
- `σy`: observation noise
"""
mutable struct SquaredExponentialKernel <: Kernel 
    l::Float64
    σf::Float64
    σy::Float64
end

function (kernel::SquaredExponentialKernel)(xp, xq, δpq)
    σf = kernel.σf
    σy = kernel.σy
    M = lengthscalematrix(kernel.l, length(xp))
    return σf^2 * exp(-0.5 * (xp - xq)' * M * (xp - xq)) + σy^2*δpq
end

lengthscalematrix(l::Float64, dimensionsize) = Diagonal(ones(dimensionsize)) / l^2

"""
    MixtureKernel(X, p, q)

A kernel mixture of Gaussian kernel and dot product.
The definition follows (6.63) in PRML.
"""
mutable struct MixtureKernel <: Kernel
    sekernel::SquaredExponentialKernel
    θ0::Float64
    θ1::Float64

    function MixtureKernel(l, σf, σy, θ0, θ1)
        sekernel = SquaredExponentialKernel(l, σf, σy)
        new(sekernel, θ0, θ1)
    end

end

function (kernel::MixtureKernel)(xp, xq, δpq)
    sekernel = kernel.sekernel
    θ0 = kernel.θ0
    θ1 = kernel.θ1
    return sekernel(xp, xq, δpq) + θ0 + θ1 * (xp' * xq)
end

function (kernel::Kernel)(inputtuple::Tuple)
    kernel(inputtuple...)
end

"""
    covarianceblock(kernel, X, Xs)

Construct covariance matrix between `X` and `Xs` based on
kernel function.

# Parameters

- `kernel`: kernel function.
- `X`: input matrix ``X``.
- `Xs`: another input matrix ``X_*`` (X-star).
- `addnoise`: if `true`, when building a covariance matrix, we add 
      observation noise to the diagonal term.
"""
function covarianceblock(kernel, X, Xs; addnoise=false)
    N = size(X, 1)
    Ns = size(Xs, 1)
    K = zeros(N, Ns)
    for p in 1:N, q in 1:Ns
        xp = X[p, :]
        xq = Xs[q, :]
        if addnoise
            K[p, q] = kernel(xp, xq, p==q) 
        else
            K[p, q] = kernel(xp, xq, false)
        end
    end
    K
end

function covariancematrix(kernel, X; addnoise=true)
    covarianceblock(kernel, X, X, addnoise=addnoise)
end

"""
Because inside squared exponential kernel we need Euclidean distance,
we can use the shortcut for it following (14.30). This can significantly 
reduce the computational cost.
"""
function covariancematrix(kernel::SquaredExponentialKernel, X)
    l, σf, σy = kernel.l, kernel.σf, kernel.σy
    N = size(X, 1)
    euclideandist = X.^2 .+ (X.^2)' .- 2*X*X'
    return σf^2 * exp.(-0.5/l^2 .* euclideandist) + Diagonal(fill(σy^2, N))
end
