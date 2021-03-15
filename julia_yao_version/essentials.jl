import Pkg;
Pkg.activate("/home/clad/Documents/Julia_Workspace/Yao")
module essentials
	using Gaston
	set(showable="svg")
	using Yao
	using StatsBase: Histogram, fit
	using BitBasis: BitStr
	Θ = LinRange(0, 2π, 100) # 50
	Φ = LinRange(0, π, 20)
	r = 1
	x = [r*cos(θ)*sin(ϕ) for θ in Θ, ϕ in Φ]
	y = [r*sin(θ)*sin(ϕ) for θ in Θ, ϕ in Φ]
	z = [r*cos(ϕ)        for θ in Θ, ϕ in Φ]
	function plot_blochsph(a::Matrix{ComplexF64})
		surf(x, y, z, lc = :turquoise,
			Axes(view = "equal xyz",
				pm3d = "depthorder",
				palette = :gray,
				style = "fill transparent solid 0.1", tics=:false, border=:false, colorbox=:false), w=:pm3d)
		surf!([0,0], [-1,1], [0,0], lc=:red)
		surf!([-1,1], [0,0], [0,0], lc=:brown)
		surf!([0,0], [0,0], [-1,1], lc=:orange)
		surf!([0,0,0,1], [0,0,-1.2,0], [1.2,-1.2,0,0.2], supp=["|0〉","|1〉","x","y"], w="labels")
		p1 = 2 * acos(abs(a[1]))
		p2 = angle(a[2]) - angle(a[1])
		x1 = r*sin(p1)*cos(p2)
		y1 = r*sin(p2)*sin(p1)
		z1 = r*cos(p1)
		surf!([0], [0], [0], supp=[[y1] [-x1] [z1]], w=:vector, lw=5, lc="'turquoise'")
	end

	function plot_hist(x::Array{BitStr{n,Int},1}) where n
		set(preamble="set xtics font ',$(n<=3 ? 15 : 15/(2^(n-3)))'")
		hist = fit(Histogram, Int.(x), 0:2^n)
		bar(hist.edges[1][1:end-1], hist.weights, fc="'dark-red'", Axes(title = :Histogram, xtics = (0:(2^n-1), "|" .* string.(0:(2^n-1), base=2, pad=n) .* "〉"), yrange = "[0:]"))	
	end
end