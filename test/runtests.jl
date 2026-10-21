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

@testset "centered_update!" begin
    # Rank-`r` data with well-separated singular values (5, 4, 3, 2, 1) about a large,
    # nonzero mean. The mean-centered matrix still has rank `r`.
    r, m, n = 5, 20, 30
    QA = Matrix(qr(randn(m, r)).Q)[:, 1:r]
    QB = Matrix(qr(randn(n, r)).Q)[:, 1:r]
    mu = 10 .+ rand(m)
    X = mu .+ QA * Diagonal(Float64.(r:-1:1)) * QB'
    Xc = X .- sum(X; dims=2) ./ n
    F = svd(Xc)

    function centered_isvd(X, nc, blocks)
        U, s, mean, k = zeros(size(X, 1), nc), zeros(nc), zeros(size(X, 1)), 0
        for b in blocks
            U, s, k = ISVD.centered_update!(U, s, mean, k, X[:, k+1:k+b])
        end
        @test k == size(X, 2)
        return U, s, mean
    end

    # Equal and unequal blocks, including single-column blocks (whose centered part is zero)
    # arriving before the data's rank has been reached
    for blocks in (fill(r, 6), [4, 1, 9, 2, 7, 7], [1, 5, 5, 5, 5, 5, 4])
        U, s, mean = centered_isvd(X, r, blocks)
        @test mean ≈ vec(sum(X; dims=2)) ./ n
        @test s ≈ F.S[1:r]
        for (col1, col2) in zip(eachcol(U), eachcol(F.U[:, 1:r]))
            @test col1 ≈ col2 || col1 ≈ -col2
        end
        # Retaining more components than the rank: the extra singular values are zero
        U, s, mean = centered_isvd(X, r+2, blocks)
        @test s[1:r] ≈ F.S[1:r]
        @test abs.(U[:, 1:r]'*F.U[:, 1:r]) ≈ I
        @test norm(s[r+1:end]) < 1e-10 * s[1]
    end

    # Cache reuse gives identical results
    cache = ISVD.Cache{Float64}(m, r, r)
    U1, s1, mean1, k1 = zeros(m, r), zeros(r), zeros(m), 0
    U2, s2, mean2, k2 = zeros(m, r), zeros(r), zeros(m), 0
    for j = 1:r:n
        U1, s1, k1 = ISVD.centered_update!(U1, s1, mean1, k1, X[:, j:j+r-1], cache)
        U2, s2, k2 = ISVD.centered_update!(U2, s2, mean2, k2, X[:, j:j+r-1])
    end
    @test U1 == U2 && s1 == s2 && mean1 == mean2 && k1 == k2
    @test_throws DimensionMismatch ISVD.centered_update!(U1, s1, mean1, k1, X[:, 1:r], ISVD.Cache{Float64}(m, r, r+1))

    # Truncation: the total centered sum of squares is kept exactly by `centered_scatter`
    Y = 3 .+ randn(m, n)
    U, s, mean, k = zeros(m, 3), zeros(3), zeros(m), 0
    total = 0.0
    for b in (7, 1, 12, 10)
        A = Y[:, k+1:k+b]
        total += ISVD.centered_scatter(A, mean, k)
        U, s, k = ISVD.centered_update!(U, s, mean, k, A)
    end
    @test total ≈ sum(abs2, Y .- sum(Y; dims=2) ./ n)
    @test mean ≈ vec(sum(Y; dims=2)) ./ n
    @test sum(abs2, s) < total

    # Invalid input is rejected without modifying the state
    U, s, mean = randn(m, r), rand(r), randn(m)
    U0, s0, mean0 = copy(U), copy(s), copy(mean)
    A = randn(m, r)
    A[2, 3] = NaN
    @test_throws ArgumentError ISVD.centered_update!(U, s, mean, 10, A)
    @test_throws ArgumentError ISVD.centered_update!(U, s, mean, 10, zeros(m, 0), ISVD.Cache{Float64}(m, r, 1))
    @test_throws DimensionMismatch ISVD.centered_update!(U, s, mean, 10, randn(m, m+1))
    @test_throws ArgumentError ISVD.centered_update!(U, s, mean, -1, randn(m, r))
    @test_throws DimensionMismatch ISVD.centered_update!(U, s, randn(m+1), 10, randn(m, r))
    @test_throws DimensionMismatch ISVD.centered_update!(U, rand(r+1), mean, 10, randn(m, r))
    @test U == U0 && s == s0 && mean == mean0
    @test_throws ArgumentError ISVD.centered_scatter(zeros(m, 0), mean, 10)
end

@testset "allocation-free centered_update!" begin
    r, m, b = 6, 100, 6
    for T in (Float64, Float32)
        X = T(5) .+ randn(T, m, 3b)
        U, s, mean = zeros(T, m, r), zeros(T, r), zeros(T, m)
        cache = ISVD.Cache{T}(m, r, b)
        A = X[:, 1:b]
        ISVD.centered_update!(U, s, mean, 0, A, cache)    # compile
        ISVD.centered_update!(U, s, mean, b, A, cache)    # warm
        @test (@allocated ISVD.centered_update!(U, s, mean, 2b, A, cache)) <= 64
    end
end
