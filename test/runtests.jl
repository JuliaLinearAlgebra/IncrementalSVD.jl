using Test, LinearAlgebra
using IncrementalSVD: IncrementalSVD as ISVD, isvd

# Test the function in the README
function isvd_with_Vt(X::AbstractMatrix{<:Real}, nc)
    Base.require_one_based_indexing(X)
    T = float(eltype(X))
    U = s = nothing
    Vt = zeros(T, nc, size(X, 2))
    cache = ISVD.Cache{T}(size(X,1), nc, nc)
    for j = 1:nc:size(X,2)
        Xchunk = @view(X[:,j:min(j+nc-1,end)])
        U, s = ISVD.update!(U, s, Xchunk, size(Xchunk, 2) == nc ? cache : nothing)
        Vt[:,j:min(j+nc-1,end)] = Diagonal(s) \ (U' * Xchunk)
    end
    return U, s, Vt
end

@testset "ISVD" begin
    r = 5
    # Build a rank-`r` target `M = A*B` with well-separated singular values
    # (5, 4, 3, 2, 1). With a poorly-conditioned target the trailing singular
    # vector is only weakly determined, and independent incremental SVD
    # computations can disagree on it by more than the default tolerance.
    QA = Matrix(qr(randn(20, r)).Q)[:, 1:r]
    QB = Matrix(qr(randn(30, r)).Q)[:, 1:r]
    A = QA * Diagonal(Float64.(r:-1:1))
    B = QB'
    M = A*B
    U = s = nothing
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1])
    end

    # Test that A is within the span of U
    pA = U*(U'*A)
    @test pA ≈ A

    # Compare with `isvd`. Singular vectors are determined only up to sign, so
    # compare columnwise allowing a flip.
    U2, s2 = isvd(M, r)
    for (col1, col2) in zip(eachcol(U), eachcol(U2))
        @test col1 ≈ col2 || col1 ≈ -col2
    end
    @test s2 ≈ s

    # Check online calculation of Vt
    Vt = Diagonal(s) \ (U' * M)
    U2, s2, Vt2 = isvd_with_Vt(M, r)
    for (col1, col2) in zip(eachcol(U), eachcol(U2))
        @test col1 ≈ col2 || col1 ≈ -col2
    end
    @test s2 ≈ s
    # A sign flip in a column of `U` flips the corresponding row of `Vt`.
    @test abs.(Vt2[:,end-r+1:end]) ≈ abs.(Vt[:,end-r+1:end])
    @test norm(M - U*Diagonal(s)*Vt) <= norm(M - U2*Diagonal(s2)*Vt2)

    # Unequal-sized blocks
    b = 4
    Ub, sb = zero(U), zero(s)
    for j = 1:b:size(M,2)
        Ub, sb = ISVD.update!(Ub, sb, M[:,j:min(j+b-1, end)])
    end
    for (col1, col2) in zip(eachcol(U), eachcol(Ub))
        @test col1 ≈ col2 || col1 ≈ -col2
    end
    @test sb ≈ s
    # Also check that `isvd` works with mismatched block sizes
    Ub2, sb2 = isvd(M, r+2)
    for (col1, col2) in zip(eachcol(Ub), eachcol(Ub2[:,1:r]))
        @test col1 ≈ col2 || col1 ≈ -col2
    end
    @test sb2[1:r] ≈ s

    # Test that two passes through `M` is like `svd([M M])`
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1])
    end
    U1, s1, _ = svd(hcat(M, M))
    @test s ≈ s1[1:r]
    @test abs.(U1[:,1:r]'*U) ≈ I

    # Mismatched element types
    U = zeros(Float32, size(A,1), r)
    s = zeros(Float32, r)
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1])
    end
    @test eltype(U) == Float32

    # NaN imputation
    M[3,8] = NaN  # imputation "works" starting with the 2nd block
    U = s = nothing
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1])
    end
    @test !any(isnan, U)
    @test !any(isnan, s)
    pA = U*(U'*A)
    @test pA ≈ A

    # Test that first-block NaNs don't break things catastrophically
    M[5,2] = NaN
    U = s = nothing
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1])
    end
    @test !any(isnan, U)
    @test !any(isnan, s)

    # Cache reuse
    M = A*B
    cache = ISVD.Cache{Float64}(size(M, 1), r, r)
    U = s = nothing
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1], cache)
    end
    U1, s1, _ = svd(M)
    @test s ≈ s1[1:r]
    @test abs.(U1[:,1:r]'*U) ≈ I
end

@testset "allocation-free update!" begin
    # `update!` with a reused `Cache` should not allocate a problem-sized
    # buffer per call: the LAPACK workspace is cached (issue #29). A small
    # constant (a `Ref` inside FastLapackInterface's `gesvd!`) is tolerated.
    r, m, n = 6, 100, 60
    QU = Matrix(qr(randn(m, r)).Q)[:, 1:r]
    QV = Matrix(qr(randn(n, r)).Q)[:, 1:r]
    M = QU * Diagonal(Float64.(r:-1:1)) * QV'
    for T in (Float64, Float32)
        MT = T.(M)
        U = zeros(T, m, r)
        s = zeros(T, r)
        cache = ISVD.Cache{T}(m, r, r)
        chunk = Matrix(MT[:, 1:r])
        ISVD.update!(U, s, chunk, cache)   # compile
        ISVD.update!(U, s, chunk, cache)   # warm
        @test (@allocated ISVD.update!(U, s, chunk, cache)) <= 64
    end
end
