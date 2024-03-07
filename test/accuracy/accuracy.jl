using Makie   # pick GLMakie or CairoMakie externally
using ISVD
using LinearAlgebra
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

function isvd_err(X, r, rextra)
    U, s = isvd(X, r+rextra)
    U = U[:,1:r]
    s = s[1:r]
    Vt = Diagonal(s) \ (U' * X)
    errisvd = norm(X - U*Diagonal(s)*Vt)
    F = svd(X)
    (; U, S, Vt) = F
    U, s, Vt = U[:,1:r], S[1:r], Vt[1:r,:]
    errsvd = norm(X - U*Diagonal(s)*Vt)
    return (errisvd - errsvd) / norm(s)
end

d = 100
r = 5
βs = 10 .^ range(-0.5, 0, length=51)
rextras = Base.IdentityUnitRange(0:10)
errmtrx = zeros(rextras, length(βs))
for (i, β) in pairs(βs)
    W = randspectrum(d, β)
    X = sample(W, 10*d)
    for rextra in rextras
        errmtrx[rextra, i] = isvd_err(X, r, rextra)
    end
end
errmtrx = max.(errmtrx, 1e-16)

fracextra = reverse(convert(Vector, OffsetArrays.no_offset_view(rextras / r)))
ploterr = reverse(OffsetArrays.no_offset_view(errmtrx); dims=1)

fig = Figure()
ax = Axis(fig[1, 1]; ylabel="β", xlabel="fraction extra components", title="Relative error of isvd")
hm = heatmap!(ax, fracextra, βs, ploterr, colorscale=log10)
Colorbar(fig[1,2], hm, label="relative error")
fig
