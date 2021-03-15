### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ feadb782-9905-4e05-a0c9-e1b1b54fb3b0
begin
	using Distributions
	n_samples = 100
	p_1 = 0.2
	x_data = rand(Binomial(1, p_1), n_samples)
end

# ╔═╡ c54da392-7d0b-11eb-278e-b3a7ec910fc7
begin
	using Gaston; set(showable="svg")
	p0 = range(0, 1;length=100)
	p1 = 1 .- p0
	plot(p0, p1, Axes(xrange = (-1, 1), yrange = (-1, 1), zeroaxis="", xtics="axis", ytics="axis"))
	plot!([0.1,1], [1,0.2], supp=["p₀","p₁"], w="labels")
end

# ╔═╡ e42579e4-5e98-4cca-a425-a3b356dd03ba
begin
	using LinearAlgebra
	p = [0.8, 0.2]
	norm(p, 1)
end

# ╔═╡ 6de3cf91-1739-49ac-aac5-0dcd72871461
begin
	using Yao
	q = zero_state(1)
end

# ╔═╡ 0ef307d4-84d3-11eb-10e7-07cc58485fba
begin
	include("./essentials.jl"); import .essentials
	essentials.plot_blochsph(state(zero_state(1)))
end

# ╔═╡ 4f69ca95-0111-40d3-b74b-34e999f4f185
md"""
Probability theory is a cornerstone for machine learning. We can think of quantum states as probability distributions with certain properties that make them different from our classical notion of probabilities. Contrasting these properties is an easy and straightforward introduction to the most basic concepts we need in quantum computing.

Apart from probability theory, linear algebra is also critical for many learning protocols. As we will see, geometry and probabilities are intrinsically linked in quantum computing, but geometric notions are also familiar in dealing with classical probability distributions. This notebook first talks about classical probabilities and stochastic vectors, and introduces quantum states as a natural generalization.

Throughout this course, we will assume finite probability distributions and finite dimensional spaces. This significantly simplifies notation and most quantum computers operate over finite dimensional spaces, so we do not lose much in generality.


# Classical probability distributions

Let us toss a biased coin. Without getting too technical, we can associate a random variable $X$ with the output: it takes the value 0 for heads and the value 1 for tails. We get heads with probability $P(X=0) = p_0$ and tails with $P(X=1) = p_1$ for each toss of the coin. In classical, Kolmogorovian probability theory, $p_i\geq 0$ for all $i$, and the probabilities sum to one: $\sum_i p_i = 1$. Let's sample this distribution
"""

# ╔═╡ c3898a67-d4b2-46a0-b5cc-69420762cd62
md"""
We naturally expect that the empirically observed frequencies also sum to one:
"""

# ╔═╡ 02a05ce8-35ad-4192-a9d6-e7d3cc29e6ba
begin
	frequency_of_zeros, frequency_of_ones = 0, 0
	for x in x_data
	    if x==1
	        frequency_of_ones += 1/n_samples
	    else
    	    frequency_of_zeros += 1/n_samples
		end
	end
	frequency_of_ones+frequency_of_zeros
end

# ╔═╡ a776d9b8-9adb-4985-9830-165acb8d805b
md"""
Since $p_0$ and $p_1$ must be non-negative, all possible probability distributions are restricted to the positive orthant. The normalization constraint puts every possible distribution on a straight line. This plot describes all possible probability distributions by biased and unbiased coins.
"""

# ╔═╡ 8c7034ce-8286-4d06-9fe1-17581695219c
md"""
We may also arrange the probabilities in a vector $\vec{p} = \begin{bmatrix} p_0 \\ p_1 \end{bmatrix}$. Here, for notational convenience, we put an arrow above the variable representing the vector, to distinguish it from scalars. You will see that quantum states also have a standard notation that provides convenience, but goes much further in usefulness than the humble arrow here.

A vector representing a probability distribution is called a *stochastic vector*. The normalization constraint essentially says that the norm of the vector is restricted to one in the $l_1$ norm. In other words, $||\vec{p}||_1 = \sum_i |p_i| = 1$. This would be the unit circle in the $l_1$ norm, but since $p_i\geq 0$, we are restricted to a quarter of the unit circle, just as we plotted above. We can easily verify this with numpy's norm function:
"""

# ╔═╡ 13533ef2-3be4-4aea-981d-0bc77ec5db57
md"""
We know that the probability of heads is just the first element in the $\vec{p}$, but since it is a vector, we could use linear algebra to extract it. Geometrically, it means that we project the vector to the first axis. This projection is described by the matrix $\begin{bmatrix} 1 & 0\\0 & 0\end{bmatrix}$. The length in the $l_1$ norm gives the sought probability:
"""

# ╔═╡ 0d931337-ad9b-45d1-85fa-2b89f052099d
begin
	Π₀ = [1 0;0 0]
	norm(Π₀ * p)
end

# ╔═╡ 29c91dc7-cfda-42d4-bd0f-f479c611339e
md"""
We can repeat the process to get the probability of tails:
"""

# ╔═╡ c1ec3e28-93fc-42d5-8ef7-454cdb7f7c33
begin
	Π₁ = [0 0; 0 1]
	norm(Π₁ * p)
end

# ╔═╡ de543889-d698-4926-a2a2-9aea48578fd8
md"""
The two projections play an equivalent role to the values 0 and 1 when we defined the probability distribution. In fact, we could define a new random variable called $\Pi$ that can take the projections $\Pi_0$ and $\Pi_1$ as values and we would end up with an identical probability distribution. This may sound convoluted and unnatural, but the measurement in quantum mechanics is essentially a random variable that takes operator values, such as projections.
"""

# ╔═╡ caa23a10-d43e-4a8c-b987-af69b55ed16e
md"""
What happens when we want to transform a probability distribution to another one? For instance, to change the bias of a coin, or to describe the transition of a Markov chain. Since the probability distribution is also a stochastic vector, we can apply a matrix on the vector, where the matrix has to fulfill certain conditions. A left *stochastic matrix* will map stochastic vectors to stochastic vectors when multiplied from the left: its columns add up to one. In other words, it maps probability distributions to probability distributions. For example, starting with a unbiased coin, the map $M$ will transform the distribution to a biased coin:
"""

# ╔═╡ cac5a2c0-6913-4997-b29d-154250be44dd
let
	p = [0.5, 0.5]
	M = [0.7 0.6; 0.3 0.4]
	norm(M * p)
end

# ╔═╡ 1dee601c-2f9f-46d8-be37-99d31e589ebe
md"""
One last concept that will come handy is entropy. A probability distribution's entropy is defined as $H(p) = - \sum_i p_i \log_2 p_i$. Let us plot it over all possible probability distributions of coin tosses:
"""

# ╔═╡ 15416de9-5fe1-4cb7-a5bb-b4227fd9ad4a
let
	ϵ = 10^-10
	p_0 = range(ϵ, 1-ϵ;length = 100)
	p_1 = 1 .- p_0
	H = -1 .* (p_0 .* log.(2,p_0) .+ p_1 .* log.(2, p_1))
	plot(p_0, H, Axes(ylabel="'H'", xlabel="'p₀'", xrange = (0,1), yrange = (0,1)))
	plot!([0.5, 0.5], [0,1], ls = :dash)
end

# ╔═╡ d803bf96-6cd4-4874-a3f7-ed7c0bf16267
md"""
Here we can see that the entropy is maximal for the unbiased coin. This is true in general: the entropy peaks for the uniform distribution. In a sense, this is the most unpredictable distribution: if we get heads with probability 0.2, betting tails is a great idea. On the other hand, if the coin is unbiased, then a deterministic strategy is of little help in winning. Entropy quantifies this notion of surprise and unpredictability.
"""

# ╔═╡ 4cf8cd38-ac26-48f5-8507-9ac56fb90867
md"""
# Quantum states

A classical coin is a two-level system: it is either heads or tails. At a first look a quantum state is a probability distribution, and the simplest case is a two-level state, which we call a qubit. Just like the way we can write the probability distribution as a column vector, we can write a quantum state as a column vector. For notational convenience that will become apparent later, we write the label of a quantum state in what is called a ket in the Dirac notation. So for instance, for some qubit, we can write 

``|\psi\rangle = \begin{bmatrix} a_0 \\ a_1 \\ \end{bmatrix}.``

In other words, a ket is just a column vector, exactly like the stochastic vector in the classical case. Instead of putting an arrow over the name of the variable to express that it is a vector, we use the ket to say that it is a column vector that represents a quantum state. There's more to this notation, as we will see.

The key difference to classical probability distributions and stochastic vectors is the normalization constraint. The square sum of their absolute values adds up to 1:

$\sqrt{|a_0|^2+|a_1|^2}=1,$

where $a_0, a_1\in \mathbb{C}$. In other words, we are normalizing in the $l_2$ norm instead of the $l_1$ norm. Furthermore, we are no longer restricted to the positive orthant: the components of the quantum state vector, which we call *probability amplitudes*, are complex valued.

Let us introduce two special qubits, corresponding to the canonical basis vectors in two dimensions: $|0\rangle$ and $|1\rangle$.

$|0\rangle = \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix}, \,\,\, |1\rangle = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}.$

This basis is also called the computational basis in quantum computing.

We can expand an arbitrary qubit state in this basis:

$|\psi\rangle = \begin{bmatrix} a_0 \\ a_1 \\ \end{bmatrix}=a_0\begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} + a_1 \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}= a_0|0\rangle+a_1|1\rangle.$

This expansion in a basis is called a superposition. If we sample the qubit state, we obtain the outcome 0 with probability $|a_0|^2$, and 1 with probability $|a_1|^2$. This is known as the Born rule; you will learn more about measurements and this rule in a subsequent notebook.

For now, let's take a look at how we can simulate classical coin tossing on a quantum computer. Let's start with a completely biased case where we get heads with probability 1. This means that our qubit $|\psi\rangle=|0\rangle$. We create a circuit of a single qubit and a single classical register where the results of the sampling (measurements) go.
"""

# ╔═╡ 1b623852-7074-4211-bbcd-a65ac55c3070
md"""
Any qubit is initialized in $|0\rangle$, so if we measure it right away, we should get our maximally biased coin.
"""

# ╔═╡ 9e2a3016-ac1a-447c-a223-3c4b80c62025
measure(q)

# ╔═╡ 3e692d90-58ce-4562-a3fa-4d73e24b7b62
md"""
Let us execute it a hundred times and study the result
"""

# ╔═╡ ee624636-c6fa-434a-bd6a-7048e755dffa
result = measure(q, nshots=100)

# ╔═╡ 0ebecb30-7db3-11eb-0dab-6ba6b1b2522a
begin
	using StatsBase
	[fit(Histogram, Int.(result), 0:2).weights string.(0:1)]
end

# ╔═╡ d6fc78fa-ba2e-45bf-af00-45fee2fd017e
md"""
As expected, all of our outcomes are $0$. 
To understand the possible quantum states, we use the Bloch sphere visualization. Since the probability amplitudes are complex and there are two of them for a single qubit, this would require a four-dimensional space. Now since the vectors are normalized, this removes a degree of freedom, allowing a three-dimensional representation with an appropriate embedding. This embedding is the Bloch sphere. It is slightly different than an ordinary sphere in three dimensions: we identify the north pole with the state $|0\rangle$, and the south pole with $|1\rangle$. In other words, two orthogonal vectors appear as if they were on the same axis -- the axis Z. The computational basis is just one basis: the axes X and Y represent two other bases. Any point on the surface of this sphere is a valid quantum state. This is also true the other way around: every pure quantum state is a point on the Bloch sphere. Here it 'pure' is an important technical term and it essentially means that the state is described by a ket (column vector). Later in the course we will see other states called mix states that are not described by a ket (you will see later that these are inside the Bloch sphere).

To make it less abstract, let's plot our $|0\rangle$ on the Bloch sphere:
"""

# ╔═╡ 63946ad1-d28d-4759-8898-085f189e57db
md"""
Compare this sphere with the straight line in the positive orthant that describes all classical probability distributions of coin tosses. You can already see that there is a much richer structure in the quantum probability space.

Let us pick another point on the Bloch sphere, that is, another distribution. Let's transform the state $|0\rangle$ to $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$. This corresponds to the unbiased coin, since we will get 0 with probability $|\frac{1}{\sqrt{2}}|^2=1/2$, and the other way around. There are many ways to do this transformation. We pick a rotation around the Y axis by $\pi/2$, which corresponds to the matrix $\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1\\1 & 1\end{bmatrix}$.
"""

# ╔═╡ a9bfa0e3-b128-43b1-af54-579d2bdcb3b7
essentials.plot_hist(measure(zero_state(1) |> Ry(π/2), nshots=1024))

# ╔═╡ 0ee44bd4-e41f-4977-b71c-9bc04bba0385
md"""
To get an intuition why it is called a rotation around the Y axis, let's plot it on the Bloch sphere:
"""

# ╔═╡ 770ae76e-6521-4613-b781-3627847ffae9
essentials.plot_blochsph(state(zero_state(1) |> Ry(π/2)))

# ╔═╡ 85bea7c1-71ba-4b6b-9341-affd291c0a7c
md"""
It does exactly what it says: it rotates from the north pole of the Bloch sphere.

Why is interesting to have complex probability amplitudes instead of non-negative real numbers? To get some insight, take a look what happens if we apply the same rotation to $|1\rangle$. To achieve this, first we flip $|0\rangle$ to $|1\rangle$ by applying a NOT gate (denoted by X in quantum computing) and then the rotation.
"""

# ╔═╡ a2eaec0e-63a1-4771-899e-240289173b90
essentials.plot_blochsph(state(zero_state(1) |> chain(1, put(1=>X), put(1=>Ry(π/2)))))

# ╔═╡ d054d2e6-da5d-4070-8b93-c81f07af756d
md"""
You can verify that the result is $\frac{1}{\sqrt{2}}(-|0\rangle + |1\rangle)$. That is, the exact same state as before, except that the first term got a minus sign: it is a negative probability amplitude. Note that the difference cannot be observed from the statistics:
"""

# ╔═╡ 0c6bdade-da51-4442-90d4-f3c36aa06e4c
begin
	essentials.plot_hist(measure(zero_state(1) |> chain(1, put(1=>X), put(1=>Ry(π/2))), nshots=1024))
end

# ╔═╡ ca2b7743-b2b3-4335-8c4c-2647492072ed
md"""
It still looks like an approximately unbiased coin. Yet, that negative sign -- or any complex value -- is what models *interference*, a critically important phenomenon where probability amplitudes can interact in a constructive or a destructive way. To see this, if we apply the rotation twice in a row on $|0\rangle$, we get another deterministic output, $|1\rangle$, although in between the two, it was some superposition. 
"""

# ╔═╡ 21a24eb3-6104-4f6a-bae5-d11e504aad62
essentials.plot_hist(measure(zero_state(1) |> chain(1, put(1=>Ry(π/2)), put(1=>Ry(π/2))), nshots=1024))

# ╔═╡ 5846d249-d73f-46d7-a1b9-09efc42ef331
md"""
Many quantum algorithms exploit interference, for instance, the seminal [Deutsch-Josza algorithm](https://en.wikipedia.org/wiki/Deutsch–Jozsa_algorithm), which is among the simplest to understand its significance.
"""

# ╔═╡ 2994d86d-c2ba-417b-aa0e-b0c7d0956008
md"""
# More qubits and entanglement

We have already seen that quantum states are probability distributions normed to 1 in the $l_2$ norm and we got a first peek at interference. If we introduce more qubits, we see another crucial quantum effect emerging. To do that, we first have to define how we write down the column vector for describing two qubits. We use a tensor product, which, in the case of qubits, is equivalent to the Kronecker product. Given two qubits, $|\psi\rangle=\begin{bmatrix}a_0\\a_1\end{bmatrix}$ and $|\psi'\rangle=\begin{bmatrix}b_0\\b_1\end{bmatrix}$, their product is $|\psi\rangle\otimes|\psi'\rangle=\begin{bmatrix}a_0b_0\\ a_0b_1\\ a_1b_0\\ a_1b_1\end{bmatrix}$. Imagine that you have two registers $q_0$ and $q_1$, each can hold a qubit, and both qubits are in the state $|0\rangle$. Then this composite state would be described by according to this product rule as follows:
"""

# ╔═╡ f98f70cf-aaf2-4710-9ef7-33c401b10c06
let
	q0 = [1, 0]
	q1 = [1, 0]
	kron(q0, q1)
end

# ╔═╡ 2f1a3452-a122-41f2-8ab7-df4e1cc01b02
md"""
This is the $|0\rangle\otimes|0\rangle$ state, which we often abbreviate as $|00\rangle$. The states $|01\rangle$, $|10\rangle$, and $|11\rangle$ are defined analogously, and the four of them give the canonical basis of the four dimensional complex space, $\mathbb{C}^2\otimes\mathbb{C}^2$.

Now comes the interesting and counter-intuitive part. In machine learning, we also work with high-dimensional spaces, but we never construct it as a tensor product: it is typically $\mathbb{R}^d$ for some dimension $d$. The interesting part of writing the high-dimensional space as a tensor product is that not all vectors in can be written as a product of vectors in the component space.

Take the following state: $|\phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$. This vector is clearly in $\mathbb{C}^2\otimes\mathbb{C}^2$, since it is a linear combination of two of the basis vector in this space. Yet, it cannot be written as $|\psi\rangle\otimes|\psi'\rangle$ for some $|\psi\rangle$, $|\psi'\rangle\in\mathbb{C}^2$.

To see this, assume that it can be written in this form. Then

$|\phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle) = \begin{bmatrix}a_0b_0\\ a_0b_1\\ a_1b_0\\ a_1b_1\end{bmatrix} = a_0b_0|00\rangle + a_0b_1|01\rangle + a_1b_0|10\rangle + a_1b_1|11\rangle.$

$|01\rangle$ and $|10\rangle$ do not appear on the left-hand side, so their coefficients must be zero: $a_1b_0=0$ and $a_0b_1=0$. This leads to a contradiction, since $a_1$ cannot be zero ($a_1b_1=1$), so $b_0$ must be zero, but $a_0b_0=1$. Therefore $|\phi^+\rangle$ cannot be written as a product.

States that cannot be written as a product are called entangled states. This is the mathematical form of describing a phenomenon of strong correlations between random variables that exceed what is possible classically. Entanglement plays a central role in countless quantum algorithms. A simple example is [quantum teleportation](https://en.wikipedia.org/wiki/Quantum_teleportation). We will also see its applications in quantum machine learning protocols.

We will have a closer look at entanglement in a subsequent notebook on measurements, but as a teaser, let us look at the measurement statistics of the $|\phi^+\rangle$ state. The explanation of the circuit preparing it will also come in a subsequent notebook.
"""

# ╔═╡ a7cdf304-48d2-4485-b473-dd2248bb7ce6
essentials.plot_hist(measure(zero_state(2) |> chain(2, put(1=>H), control(1, 2=>X)), nshots=1024))

# ╔═╡ 5a2944be-209c-4fbe-b1b1-19a5f9df4316
md"""
Notice that 01 or 10 never appear in the measurement statistics.
"""

# ╔═╡ ec3512fa-22de-4123-bb1f-440b899ac901
md"""
# Further reading

Chapter 9 in Quantum Computing since Democritus by Scott Aaronson describes a similar approach to understanding quantum states -- in fact, the interference example was lifted from there.
"""

# ╔═╡ Cell order:
# ╟─4f69ca95-0111-40d3-b74b-34e999f4f185
# ╠═feadb782-9905-4e05-a0c9-e1b1b54fb3b0
# ╟─c3898a67-d4b2-46a0-b5cc-69420762cd62
# ╠═02a05ce8-35ad-4192-a9d6-e7d3cc29e6ba
# ╟─a776d9b8-9adb-4985-9830-165acb8d805b
# ╠═c54da392-7d0b-11eb-278e-b3a7ec910fc7
# ╟─8c7034ce-8286-4d06-9fe1-17581695219c
# ╠═e42579e4-5e98-4cca-a425-a3b356dd03ba
# ╟─13533ef2-3be4-4aea-981d-0bc77ec5db57
# ╠═0d931337-ad9b-45d1-85fa-2b89f052099d
# ╟─29c91dc7-cfda-42d4-bd0f-f479c611339e
# ╠═c1ec3e28-93fc-42d5-8ef7-454cdb7f7c33
# ╟─de543889-d698-4926-a2a2-9aea48578fd8
# ╟─caa23a10-d43e-4a8c-b987-af69b55ed16e
# ╠═cac5a2c0-6913-4997-b29d-154250be44dd
# ╟─1dee601c-2f9f-46d8-be37-99d31e589ebe
# ╠═15416de9-5fe1-4cb7-a5bb-b4227fd9ad4a
# ╟─d803bf96-6cd4-4874-a3f7-ed7c0bf16267
# ╟─4cf8cd38-ac26-48f5-8507-9ac56fb90867
# ╠═6de3cf91-1739-49ac-aac5-0dcd72871461
# ╟─1b623852-7074-4211-bbcd-a65ac55c3070
# ╠═9e2a3016-ac1a-447c-a223-3c4b80c62025
# ╟─3e692d90-58ce-4562-a3fa-4d73e24b7b62
# ╠═ee624636-c6fa-434a-bd6a-7048e755dffa
# ╠═0ebecb30-7db3-11eb-0dab-6ba6b1b2522a
# ╟─d6fc78fa-ba2e-45bf-af00-45fee2fd017e
# ╠═0ef307d4-84d3-11eb-10e7-07cc58485fba
# ╟─63946ad1-d28d-4759-8898-085f189e57db
# ╠═a9bfa0e3-b128-43b1-af54-579d2bdcb3b7
# ╟─0ee44bd4-e41f-4977-b71c-9bc04bba0385
# ╠═770ae76e-6521-4613-b781-3627847ffae9
# ╟─85bea7c1-71ba-4b6b-9341-affd291c0a7c
# ╠═a2eaec0e-63a1-4771-899e-240289173b90
# ╟─d054d2e6-da5d-4070-8b93-c81f07af756d
# ╠═0c6bdade-da51-4442-90d4-f3c36aa06e4c
# ╟─ca2b7743-b2b3-4335-8c4c-2647492072ed
# ╠═21a24eb3-6104-4f6a-bae5-d11e504aad62
# ╟─5846d249-d73f-46d7-a1b9-09efc42ef331
# ╟─2994d86d-c2ba-417b-aa0e-b0c7d0956008
# ╠═f98f70cf-aaf2-4710-9ef7-33c401b10c06
# ╟─2f1a3452-a122-41f2-8ab7-df4e1cc01b02
# ╠═a7cdf304-48d2-4485-b473-dd2248bb7ce6
# ╟─5a2944be-209c-4fbe-b1b1-19a5f9df4316
# ╟─ec3512fa-22de-4123-bb1f-440b899ac901
