# ISVD

ISVD provides incremental singular value decomposition, as described in

> Brand, M. "Fast low-rank modifications of the thin singular value
> decomposition."  Linear algebra and its applications 415.1 (2006):
> 20-30.

It performs NaN-imputation as described in

> Brand, M. "Incremental singular value decomposition of uncertain
> data with missing values."  Computer Vision—ECCV 2002. Springer
> Berlin Heidelberg, 2002. 707-720.


Here's a demo in which we process `A` in chunks of 4 columns:

```julia
julia> A = randn(5, 12);

julia> using ISVD, LinearAlgebra

julia> U, s = update_U_s!(nothing, nothing, A[:,1:4]);   # use `nothing` to initialize

julia> update_U_s!(U, s, A[:, 5:8]);

julia> update_U_s!(U, s, A[:, 9:12]);

julia> s
4-element Vector{Float64}:
 4.351123559463465
 4.18050301615471
 3.662876466035874
 2.923979120208828

julia> F = svd(A);

julia> F.S
5-element Vector{Float64}:
 4.351167907836934
 4.182452959982528
 3.669488216333535
 2.9398639871271564
 1.7956053622541457
```

Currently, `update_U_s!` cannot handle a block smaller than the number of components, so unless the number of columns
of `A` is an exact multiple of the blocksize, a few final columns may be omitted.

Note that `V` may be obtained as `(Diagonal(1 ./ s) * (U \ A))'`.
