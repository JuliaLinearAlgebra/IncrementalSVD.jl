module IncrementalSVD

using LinearAlgebra

export isvd

"""
IncrementalSVD implements incremental singular value decomposition.

The public functions are:
- [`isvd`](@ref)
- [`IncrementalSVD.update!`](@ref)        (not exported)
- [`IncrementalSVD.impute_nans!`](@ref)   (not exported)
- [`IncrementalSVD.Cache`](@ref)          (not exported)
""" IncrementalSVD

"""
    U, s = isvd(X::AbstractMatrix{<:Real}, nc)

Compute an incremental thin SVD of the matrix `X`, returning the left singular
vectors `U` and the singular values `s`.  The number of retained components
is specified by `nc`.

`V` may be obtained via `V = (X' * U) / Diagonal(s)`.

`isvd` is just a wrapper around repeated calls to [`IncrementalSVD.update!`](@ref).
In cases with streaming or large `X` that cannot fit into memory, you may prefer
to use `IncrementalSVD.update!` directly with smaller chunks of `X`.
"""
function isvd(X::AbstractMatrix{<:Real}, nc)
    Base.require_one_based_indexing(X)
    T = float(eltype(X))
    U = zeros(T, size(X,1), nc)
    s = zeros(T, nc)
    cache = Cache{T}(size(X,1), nc, nc)
    for j = 1:nc:size(X,2)
        Xchunk = @view(X[:,j:min(j+nc-1,end)])
        U, s = update!(U, s, Xchunk, size(Xchunk, 2) == nc ? cache : nothing)
    end
    U, s
end

struct Cache{T}
    A::Matrix{T}      # NaN-imputation may modify values, don't destroy the input
    UtA::Matrix{T}    # U'*A
    Uperp::Matrix{T}  # A - U*UtA, the part of A orthogonal to U; will also hold P once that gets computed
    UtP::Matrix{T}    # U'*P
    Pperp::Matrix{T}  # P - U*UtP, project out U from P (needed for roundoff error)
    R::Matrix{T}      # The upper triangular matrix from the QR decomposition of Uperp
    K::Matrix{T}      # Eq. 4, Brand 2006
    UP::Matrix{T}     # [U P]
    UProt::Matrix{T}  # UP*U′, the rotated part of [U P] (the first r columns will be the new U)
    Us::Matrix{T}     # U*Diagonal(s)
end

"""
    Cache{T}(m::Int, r::Int, b::Int)
    Cache(U, A)

A cache for the incremental SVD algorithm.  This is a struct for storing
intermediate results of the incremental SVD algorithm.  It can be used to
reduce memory allocation during [`IncrementalSVD.update!`](@ref).

`m` is the number of rows in the matrix we're computing the SVD of, `r` is
the rank of the SVD we're computing, and `b` is the blocksize. Concretely,
`size(U) = (m, r)` and `size(A) = (m, b)`.
"""
Cache{T}(m::Int, r::Int, b::Int) where T = Cache{T}(
    zeros(T, m, b),     # A
    zeros(T, r, b),     # UtA
    zeros(T, m, b),     # Uperp
    zeros(T, r, b),     # UtP
    zeros(T, m, b),     # Pperp
    zeros(T, b, b),     # R
    zeros(T, r+b, r+b), # K
    zeros(T, m, r+b),   # UP
    zeros(T, m, r+b),   # UProt
    zeros(T, m, r),     # Us
)
function Cache(U::AbstractMatrix, A::AbstractMatrix)
    Base.require_one_based_indexing(U, A)
    size(U, 1) == size(A,  1) || throw(DimensionMismatch("number of rows in U and A must match, got $(size(U, 1)) and $(size(A, 1))"))
    T = typeof(oneunit(eltype(U)) * oneunit(eltype(A)))
    return Cache{T}(size(U)..., size(A, 2))
end

"""
    U, s = IncrementalSVD.update!(U, s, A, cache=Cache(U, A))

Update a thin SVD with a new matrix `A` of data, as if `A` had been appended via `hcat`
to the original matrix. `A` can be thought of as a "chunk" in an incremental
computation of the SVD. `U` and `s` are updated in-place as well as returned.
You can reuse temporary storage by creating `cache` (see [`IncrementalSVD.Cache`](@ref)).

There are two ways to initialize:
- `U, s = zeros(T, m, r), zeros(T, r)`. This specifies the element type `T`, the
  number of rows `m` and the rank `r`.
- `U, s = nothing, nothing`. This will use `size(U) = size(A)`, i.e.,
  the chunk size specifies the truncated rank.

If `A` has NaNs, replacement values will be imputed, but `A` itself will not
be modified. See [`IncrementalSVD.impute_nans!`](@ref) if you'd like to run the imputation
separately.

If you are computing only `U` and `s`, you can obtain `V` from

    Vt = Diagonal(s) \\ (U' * X)

or

    V = (X' * U) / Diagonal(s)

where `X` is the complete matrix containing all the data. Of course, this
too can be computed incrementally using a second pass through `X`.
While `V` can be computed on-the-fly, this is not recommended, as it means
the early columns of `V'` will be computed using an early approximation to `U`
and `s`. A more consistent result is obtained from a second pass through The
data.

# Extended help

NaN-imputation is performed by

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
"""
function update!(U::AbstractMatrix, s::AbstractVector, A::AbstractMatrix, cache::Cache)
    Base.require_one_based_indexing(U, s, A)
    m, r = size(U)
    mA, b = size(A)
    m == mA || throw(DimensionMismatch("number of rows in U and A must match, got $m and $mA"))
    copyto!(cache.A, A)
    mul!(cache.Us, U, Diagonal(s))  # U*S is needed for NaN-imputation
    impute_nans!(cache.A, cache.Us)
    (; UtA, Uperp, UtP, Pperp, R, K, UP, UProt) = cache
    mul!(UtA, U', cache.A)          # projection onto space spanned by U
    mul!(Uperp, U, UtA)             # projection back into U-space
    negsub!(Uperp, cache.A)         # the part of A orthogonal to U
    P, R = qrf!(Uperp, R)
    # Nominally, P is already orthogonal to U. But if the number of retained components exceeds
    # the actual rank of the matrix, this may not be true (it orthogonalizes noise).
    # So we'll project out the part of P that's in the span of U.
    mul!(UtP, U', P)
    mul!(Pperp, U, UtP)
    negsub!(Pperp, P)
    copyto!(view(UP, :, 1:r), U)    # UP = [U P]
    copyto!(view(UP, :, r+1:r+b), Pperp)
    # Because we treat `A` as new columns of data, in Brand 2006 we have B'[:,currentcolumns] = I;
    # since V' is zero over this span of columns, V'*B = 0, and thus Q = R_B = I.
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
update!(U::AbstractMatrix, s::AbstractVector, A::AbstractMatrix) = update!(U, s, A, Cache(U, A))
update!(U::AbstractMatrix, s::AbstractVector, A::AbstractMatrix, ::Nothing) = update!(U, s, A, Cache(U, A))

function update!(::Nothing, s::Nothing, A::AbstractMatrix, cache::Cache=Cache(A, A))
    copyto!(cache.A, A)
    cache.A[isnan.(A)] .= 0
    U, s, _ = svd(cache.A)
    return U, s
end

"""
    impute_nans!(A, U, s)

Impute missing values in `A` using the SVD `U` and `s`.
`A` is modified in place and returned.
"""
function impute_nans!(A, Us)
    # Impute missing values
    for j in axes(A, 2)
        c = view(A, :, j)
        nanflag = isnan.(c)
        sum(nanflag) == 0 && continue
        notnanflag = map(!, nanflag)
        c[nanflag] = Us[nanflag,:] * (Us[notnanflag,:] \ c[notnanflag])
    end
    return A
end
impute_nans!(A, U, s) = impute_nans!(A, U*Diagonal(s))

function negsub!(dest, src)
    @inbounds @simd for I in eachindex(dest, src)
        dest[I] = src[I]-dest[I]
    end
    return dest
end

# For this application, geqrf is faster
function qrf!(P, R)
    m, b = size(P)
    m >= b || throw(DimensionMismatch("Works only for m > b"))
    P, tau = LAPACK.geqrf!(P)
    fill!(R, zero(eltype(R)))
    for j = 1:b, i = 1:j
        R[i,j] = P[i,j]
    end
    LAPACK.orgqr!(P, tau)
    return P, R
end

end
