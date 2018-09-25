module ISVD

using LinearAlgebra

"""
ISVD provides incremental singular value decomposition.

Exported function: `update_U_s!`
""" ISVD

export update_U_s!

"""
```
U, s = update_U_s!(U, s, A)
```

Update a thin SVD with a new matrix `A` of data, as if `A` had been
appended via `hcat` to the original matrix.  Initialize with empty `U`
and `s`; otherwise, the sizes of `U` and `A` must match.  This keeps
the largest `size(A,2)` singular values and left-vectors.

`U` and `s` will be modified in-place except during initialization.
If `A` has NaNs, those will be destroyed by this computation---the
algorithm uses the NaN-imputation of

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

To compute `V` use `V = M'*U/S`, where `M` is the complete matrix
containing all the data.
"""
function update_U_s!(U, s, A::AbstractMatrix)
    if isempty(U)
        A[isnan.(A)] .= 0
        Unew, snew, _ = svd(A)
        return oftype(U, Unew), oftype(s, snew)
    end
    size(U) == size(A) || throw(DimensionMismatch("Size of U ($(size(U))) and A ($(size(A))) must match"))
    r = size(A, 2)
    impute_nans!(A, U, s)
    UA = U'*A                 # projection onto space spanned by U
    Uperp = U*UA
    negsub!(Uperp, A)         # the part of A orthogonal to U
    P, R = qrf!(Uperp)
    K = zeros(eltype(s), 2r, 2r)
    for j = 1:r
        K[j,j] = s[j]
    end
    K[1:r, r+1:2r] = UA
    K[r+1:2r, r+1:2r] = R
    # Up, sp, _ = svd(K)
    # For this application, gesvd is faster, partly because we can
    # skip computing V
    Up, sp, _ = LAPACK.gesvd!('O', 'N', copy(K))
    Ucat = hcat(U, P)
    copyto!(U, view(Ucat*Up, :, 1:r))
    copyto!(s, 1, sp, 1, r)
    U, s
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
function qrf!(A)
    m, n = size(A)
    m >= n || throw(DimensionMismatch("Works only for m > n"))
    A, tau = LAPACK.geqrf!(A)
    R = zeros(eltype(A), (n,n))
    for j = 1:n, i = 1:j
        R[i,j] = A[i,j]
    end
    LAPACK.orgqr!(A, tau)
    A, R
end

end
