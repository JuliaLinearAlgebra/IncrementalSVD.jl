module ISVD

using LinearAlgebra

"""
ISVD provides incremental singular value decomposition.

Exported function: `update_U_s!`
""" ISVD

"""
    Cache{T}(m::Int, r::Int, b::Int)
    Cache(U, A)

A cache for the incremental SVD algorithm.  This is a struct that holds the
intermediate results of the incremental SVD algorithm.  It is used to avoid
repeatedly allocating memory for intermediate results.

`m` is the number of rows in the matrix we're computing the SVD of, `r` is
the rank of the SVD we're computing, and `b` is the blocksize. Concretely,
`size(U) = (m, r)` and `size(A) = (m, b)`.
"""
struct Cache{T}
    A::Matrix{T}      # NaN-imputation may modify values, don't destroy the input
    UtA::Matrix{T}    # U'*A
    Uperp::Matrix{T}  # A - U*UtA, the part of A orthogonal to U; will also hold P once that gets computed
    R::Matrix{T}      # The upper triangular matrix from the QR decomposition of Uperp
    K::Matrix{T}      # Eq. 4, Brand 2006
    UP::Matrix{T}     # [U P]
    UProt::Matrix{T}  # UP*U′, the rotated part of [U P] (the first r columns will be the new U)
end
Cache{T}(m::Int, r::Int, b::Int) where T = Cache{T}(
    zeros(T, m, b),     # A
    zeros(T, r, b),     # UtA
    zeros(T, m, b),     # Uperp
    zeros(T, b, b),     # R
    zeros(T, r+b, r+b), # K
    zeros(T, m, r+b),   # UP
    zeros(T, m, r+b),   # UProt
)
function Cache(U::AbstractMatrix, A::AbstractMatrix)
    Base.require_one_based_indexing(U, A)
    size(U, 1) == size(A,  1) || throw(DimensionMismatch("number of rows in U and A must match, got $(size(U, 1)) and $(size(A, 1))"))
    T = typeof(oneunit(eltype(U)) * oneunit(eltype(A)))
    return Cache{T}(size(U)..., size(A, 2))
end

"""
    U, s = ISVD.update!(U, s, A, cache=Cache(U, A))
    U, s, V = ISVD.update!(U, s, V, A, cache=Cache(U, A))

Update a thin SVD with a new matrix `A` of data, as if `A` had been appended via `hcat`
to the original matrix. `A` can be thought of as a "chunk" in an incremental
computation of the SVD. `U`, `s`, and optionally `V` are updated in-place as well as returned.
You can reuse temporary storage by creating `cache`

There are two ways to initialize:
- `U, s, V = zeros(T, m, r), zeros(T, r), zeros(T, n, r)`. This specifies
  the element type `T`, the number of rows `m`, the rank `r`, and the number
  of columns `n`. If you're computing `V`, this is the only option.
- `U, s = nothing, nothing`. This will use `size(U) = size(A)`, i.e.,
  the chunk size specifies the truncated rank.

If `A` has NaNs, replacement values will be imputed by

> Brand, M. "Incremental singular value decomposition of uncertain
> data with missing values."  Computer Vision—ECCV 2002. Springer
> Berlin Heidelberg, 2002. 707-720.

This implements the algorithm described in section 2 of

> Brand, M. "Fast low-rank modifications of the thin singular value
> decomposition."  Linear algebra and its applications 415.1 (2006):
> 20-30.

in a manner similar to

> Baker, Christopher Grover. "A block incremental algorithm for
> computing dominant singular subspaces." (2004). (Thesis)

While the Brand paper advocates rank-1 updates, Grover points out
that the Gu & Eisenstadt "broken arrow matrix" SVD needed to make that
efficient has a very high coefficient. In testing, the block approach
seems much more efficient.

If you are computing only `U` and `s`, you can obtain `V` from

    Vt = Diagonal(s) \\ (U' * X)

or

    V = (X' * U) / Diagonal(s)

where `X` is the complete matrix containing all the data. Of course, this
too can be computed incrementally using a second pass through `X`.
"""
function update!(U::AbstractMatrix, s::AbstractVector, A::AbstractMatrix, cache::Cache=Cache(U, A))
    Base.require_one_based_indexing(U, s, A)
    m, r = size(U)
    mA, b = size(A)
    m == mA || throw(DimensionMismatch("number of rows in U and A must match, got $m and $mA"))
    copyto!(cache.A, A)
    impute_nans!(cache.A, U, s)
    (; UtA, Uperp, R, K, UP, UProt) = cache
    mul!(UtA, U', cache.A)          # projection onto space spanned by U
    mul!(Uperp, U, UtA)             # projection back into U-space
    negsub!(Uperp, cache.A)         # the part of A orthogonal to U
    P, R = qrf!(Uperp, R)
    copyto!(view(UP, :, 1:r), U)    # UP = [U P]
    copyto!(view(UP, :, r+1:r+b), P)
    fill!(K, zero(eltype(K)))       # Eq. 4, Brand 2006
    for j = 1:r
        K[j,j] = s[j]
    end
    K[1:r, r+1:r+b] = UtA
    K[r+1:r+b, r+1:r+b] = R
    # Up, sp, _ = svd(K)
    # For this application, gesvd is faster, partly because we can
    # skip computing V
    U′, s′, _ = LAPACK.gesvd!('O', 'N', K)
    mul!(UProt, UP, U′)
    copyto!(U, view(UProt, :, 1:r))
    copyto!(s, view(s′, 1:r))
    return U, s
end

function update!(::Nothing, s::Nothing, A::AbstractMatrix, cache::Cache=Cache(A, A))
    copyto!(cache.A, A)
    cache.A[isnan.(A)] .= 0
    U, s, _ = svd(cache.A)
    return U, s
end

function impute_nans!(A, U, s)
    # Impute missing values
    for j = 1:size(A,2)
        c = view(A, :, j)
        nanflag = isnan.(c)
        if sum(nanflag) == 0
            continue
        end
        notnanflag = map(!, nanflag)
        c[nanflag] = U[nanflag,:]*Diagonal(s)*(U[notnanflag,:]*Diagonal(s)\c[notnanflag])
    end
    A
end

function negsub!(dest, src)
    @inbounds @simd for I in eachindex(dest, src)
        dest[I] = src[I]-dest[I]
    end
    dest
end

# For this application, geqrf is faster
function qrf!(P, R)
    m, b = size(P)
    m >= b || throw(DimensionMismatch("Works only for m > b"))
    P, tau = LAPACK.geqrf!(P)
    for j = 1:b, i = 1:j
        R[i,j] = P[i,j]
    end
    LAPACK.orgqr!(P, tau)
    P, R
end

end
