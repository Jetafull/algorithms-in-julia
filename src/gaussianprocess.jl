using Optim: optimize, Fminbox, LBFGS
using Zygote: gradient

include("./kernels.jl")

"""
Following (15.14) in MLAPP, the posterior predictive for noise free 
observation ``\\mathbf{f}_*`` with new data ``\\mathbf{X}_*`` is

``
p(\\mathbf{f}_* | \\mathbf{X}_*, \\mathbf{X}, \\mathbf{y}) =
\\mathscr{N}(\\mathbf{f}_* | \\mathbf{\\mu}_*, \\mathbf{\\Sigma}_*)
``

To derive posterior predictive distribution for ``\\mathbf{y}_*``, 
we need to add the observation noise to ``\\mathbf{f}_*``.
"""
function predict(kernel, X, y, Xs; decomp=false)
    Ky = covariancematrix(kernel, X)
    Ks = covarianceblock(kernel, X, Xs)
    Kss = covariancematrix(kernel, Xs)
    invKy = inv(Ky)
    μs = Ks'*invKy*y
    Σs = Kss - Ks'*invKy*Ks
    return μs, Σs
end

"""
Negative log likelihood for the marginal distribution ``p(y|X)``.
The numerically stable version follows **Algorithm 15.1** in MLAPP.
"""
function negloglike(X, y, Ky; stable=false)
    N = size(X, 1)
    if stable
        L = cholesky(Ky).L
        invL = inv(L)
        α = invL'*invL*y 
        loglike = -0.5*y'*α - sum(log.(diag(L))) - 0.5*N*log(2π)
        return -loglike
    else
        logdetKy = log(det(Ky))
        loglike = -0.5*y'*inv(Ky)*y - 0.5*logdetKy - 0.5*N*log(2π)
        return -loglike
    end
end

"""
Negative log-likelihood for squared exponential kernel (log parameters).
"""
function sqexp_negloglike_log(X, y, logl, logσf, logσy; stable=false)
    l = exp(logl) 
    σf = exp(logσf)
    σy = exp(logσy)
    sekernel = SquaredExponentialKernel(l, σf, σy)
    # Ky = covariancematrix_alt(sekernel, X)
    Ky = covariancematrix(sekernel, X)
    return negloglike(X, y, Ky, stable=stable)
end

function sqexp_negloglike(X, y, l, σf, σy; stable=false)
    sekernel = SquaredExponentialKernel(l, σf, σy)
    Ky = covariancematrix(sekernel, X)
    return negloglike(X, y, Ky, stable=stable)
end

"""
Analytic gradient of the log marginal likelihood for squared exponential kernel.
"""
function sqexp_gradient_analytic!(G, X, y, paramvals...)
    sekernel = SquaredExponentialKernel(paramvals...)
    grad = -logmarginalgradient(sekernel, X, y)
    for i = 1:size(G, 1)
        G[i] = grad[i]
    end
    nothing
end

"""
Autodiff gradient of the log marginal gradient using Zygote.
"""
function sqexp_gradient_autodiff!(G, X, y, paramvals...)
    grad = gradient(params -> sqexp_negloglike(X, y, params...), paramvals)[1]
    for i = 1:size(G, 1)
        G[i] = grad[i]
    end
    nothing
end

function optimize_sekernel(
    X, y; 
    init_params=ones(3), 
    lower=[1e-6, 1e-6, 1e-6], 
    upper=[Inf, Inf, Inf],
    stable=true,
    optimizer=LBFGS()
    )
    result = optimize(
        params -> sqexp_negloglike(X, y, params..., stable=stable),
        lower, 
        upper, 
        init_params,
        Fminbox(optimizer)
    )
    return result
end

"""
Optimizing squared exponential kernel hyperparameters with provided gradient.
The second input to `optimize` is the function to calculate gradient.
Its first argument is inserted by the `optimize` function to keep track
of the current gradient.
"""
function optimize_sekernel(
    X, y, g!; 
    init_params=ones(3), 
    lower=[1e-6, 1e-6, 1e-6], 
    upper=[Inf, Inf, Inf],
    stable=true,
    optimizer=LBFGS()
    )
    result = optimize(
        params -> sqexp_negloglike(X, y, params..., stable=stable),
        (G, params) -> g!(G, X, y, params...),
        lower, 
        upper, 
        init_params,
        Fminbox(optimizer)
    )
    return result
end

function mixture_negloglike(X, y, logl, logσf, logσy, θ0, θ1)
    l = exp(logl) 
    σf = exp(logσf)
    σy = exp(logσy)
    mixturekernel = MixtureKernel(l, σf, σy, θ0, θ1)
    Ky = covariancematrix(mixturekernel, X)
    return negloglike(X, y, Ky)
end

"""
    logmarginalpartial(kernel, X, y, ∂Ky∂θ)

Partial derivative of log marginal likelihood ``\\log{P(y|X)}`` wrt
parameter ``θ``.

# Parameters
- `X`: input matrix
- `y`: labels
- `∂Ky∂θ`: partial derivative of covariance matrix ``K_y`` wrt ``θ``.

"""
function logmarginalpartial(kernel, X, y, ∂Ky∂θ) 
    Ky = covariancematrix(kernel, X)
    invKy = inv(Ky)
    α = invKy*y
    return 0.5*tr((α*α' - invKy)*∂Ky∂θ)
end

"""
    logmarginalgradient(kernel, X, y)
  
``∇_θ \\log{P(y|X)}``: Analytic gradient of log marginal likelihood ``\\log{P(y|X)}`` 
wrt parameters. 

Notice because we are passing ``\\logθ`` into the log marginal 
likelihood function. 
    
We need to apply chain rule to the gradient we need is 
``
∇_{\\logθ} \\log{P(y|X)} = \\frac{∂log{P(y|X)}}{∂θ^2} 
                         \\frac{∂θ^2}{∂θ} 
                         \\frac{∂θ}{∂logθ}  
``

# Parameters

- `kernel`: kernel function
- `X`: input matrix
- `y`: observed outcome 

"""
function logmarginalgradient(kernel::SquaredExponentialKernel, X, y)
    l = kernel.l
    σf = kernel.σf
    σy = kernel.σy

    funcs = Dict{String, Function}()
    quad(xp, xq, M) = -0.5*(xp-xq)'*M*(xp-xq)
    funcs["∂p∂logl"] = (xp, xq, δpq, M) -> 
        σf^2*(-0.5*(xp-xq)'*(xp-xq))*exp.(quad(xp, xq, M))*(-1/l^4)*2*l*l
    funcs["∂p∂logσf"] = (xp, xq, δpq, M) -> exp.(quad(xp, xq, M))*2*σf*σf
    funcs["∂p∂logσy"] = (xp, xq, δpq, M) -> δpq*2*σy*σy

    N = size(X, 1)
    numparameters = 3
    parameternames = ["∂p∂logl", "∂p∂logσf", "∂p∂logσy"]
    marginalpartialderiv = zeros(numparameters)
    for i = 1:numparameters
        parametername = parameternames[i]
        func = funcs[parametername]
        ∂Ky∂θ = zeros(N, N)
        for p=1:N, q=1:N  
            xp = X[p, :]
            xq = X[q, :]
            M = lengthscalematrix(l, length(xp))
            ∂Ky∂θ[p, q] = func(xp, xq, p==q, M)
        end
        marginalpartialderiv[i] = logmarginalpartial(kernel, X, y, ∂Ky∂θ)
    end

    marginalpartialderiv
end
