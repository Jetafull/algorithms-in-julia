### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e9146374-dd10-11ea-1b83-c978edacc525
begin
	using Plots
	using Random
	using Distributions
	using Printf
	
	pyplot();
	Random.seed!(20200813);
end

# ╔═╡ 7396bcf4-dd11-11ea-2577-55daf9dbbd68
begin
	N = 15
	noise_scale = 5
	x = collect(range(1, 10, length=N))
	ϵ = noise_scale*randn(N)
	y = x .+ x.^2 .+ ϵ
end

# ╔═╡ d32c85c0-dd13-11ea-0476-8bec5f639c3a
begin
	# loss function
	loss(w) = mean((w.*x - y).^2)
	gradient(w) = mean(2*(w.*x - y).*x)
end

# ╔═╡ 3fdce730-dd39-11ea-216b-efd542f9528e
@bind w html"<input type='range' min='-10' max='30' value='$w' step='0.1'>"

# ╔═╡ e0649a0e-dd11-11ea-1e21-19d9fb8fe64b
begin	
	η = 0.01
	current_loss = loss(w)
	current_loss_str = @sprintf "%.1f" current_loss
	w_str = @sprintf "%.2f" w
	grad = gradient(w)
	
	p1 = scatter(x, y, label="")
	ŷ = w .* x
	plot!(x, ŷ, label="ŷ")
	
	p2 = plot(loss, -10:25,  xlims=(-10, 30), ylims = (0,10000), label="loss(w)")
	annotate!((w, loss(w)*1.1, Plots.text("w=$w_str, loss=$current_loss_str", 10) ))
	vline!([w], label="")
	plot!([w, w-η*grad], fill(current_loss, 2), label="gradient", color=:red, arrow=:right)
	
	plot(p1, p2, layout=(1, 2))
	
end

# ╔═╡ Cell order:
# ╠═e9146374-dd10-11ea-1b83-c978edacc525
# ╠═7396bcf4-dd11-11ea-2577-55daf9dbbd68
# ╠═d32c85c0-dd13-11ea-0476-8bec5f639c3a
# ╠═3fdce730-dd39-11ea-216b-efd542f9528e
# ╟─e0649a0e-dd11-11ea-1e21-19d9fb8fe64b
