using Makie   # pick GLMakie or CairoMakie externally
using ISVD
using LinearAlgebra
using RandomizedLinAlg
using OffsetArrays

function randspectrum(d, β)
    Q, _ = qr(randn(d, d))
    return Q*Diagonal(sqrt.([β^i for i=0:d-1]))
end

function sample(W, n)
    d = size(W, 1)
    X = randn(d, n)
    return W*X
end

function errs(X, r, rextra)
    U, s = isvd(X, r+rextra)
    U = U[:,1:r]
    s = s[1:r]
    Vt = Diagonal(s) \ (U' * X)
    errisvd = norm(X - U*Diagonal(s)*Vt)
    U, s, V = rsvd(X, r, rextra)
    errrsvd = norm(X - U*Diagonal(s)*V')
    F = svd(X)
    (; U, S, Vt) = F
    U, s, Vt = U[:,1:r], S[1:r], Vt[1:r,:]
    errsvd = norm(X - U*Diagonal(s)*Vt)
    return (errisvd - errsvd) / norm(s), (errrsvd - errsvd) / norm(s)
end

d = 100
r = 5
βs = 10 .^ range(-0.5, 0, length=51)
rextras = Base.IdentityUnitRange(0:10)
errmtrxi = zeros(rextras, length(βs))
errmtrxr = zeros(rextras, length(βs))
for (i, β) in pairs(βs)
    W = randspectrum(d, β)
    X = sample(W, 10*d)
    for rextra in rextras
        errmtrxi[rextra, i], errmtrxr[rextra, i] = errs(X, r, rextra)
    end
end
errmtrxi = max.(errmtrxi, 1e-16)
errmtrxr = max.(errmtrxr, 1e-16)
colorrange = (1e-16, max(maximum(errmtrxi), maximum(errmtrxr)))

fracextra = reverse(convert(Vector, OffsetArrays.no_offset_view(rextras / r)))
ploterri = reverse(OffsetArrays.no_offset_view(errmtrxi); dims=1)
ploterrr = reverse(OffsetArrays.no_offset_view(errmtrxr); dims=1)

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1][1, 1]; ylabel="β", xlabel="fraction extra components", title="Relative error of isvd")
hmi = heatmap!(ax, fracextra, βs, ploterri; colorscale=log10, colorrange)
ax = Axis(fig[1, 2][1, 1]; ylabel="β", xlabel="fraction extra components", title="Relative error of rsvd")
hmr = heatmap!(ax, fracextra, βs, ploterrr; colorscale=log10, colorrange)
Colorbar(fig[1,2][1,2], hmr, label="relative error")
fig
