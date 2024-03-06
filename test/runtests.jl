using Test, LinearAlgebra
using ISVD

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
    A = rand(20, r)
    B = rand(r, 30)
    M = A*B
    U = s = nothing
    for j = 1:r:size(M,2)
        U, s = ISVD.update!(U, s, M[:,j:j+r-1])
    end

    # Test that A is within the span of U
    pA = U*(U'*A)
    @test pA ≈ A

    # Compare with `isvd`
    U2, s2 = isvd(M, r)
    @test U2 ≈ U
    @test s2 ≈ s

    # Check online calculation of Vt
    Vt = Diagonal(s) \ (U' * M)
    U2, s2, Vt2 = isvd_with_Vt(M, r)
    @test U2 ≈ U
    @test s2 ≈ s
    @test Vt2[:,end-r+1:end] ≈ Vt[:,end-r+1:end]
    @test norm(M - U*Diagonal(s)*Vt) <= norm(M - U2*Diagonal(s2)*Vt2)

    # Unequal-sized blocks
    b = 3
    Ub, sb = zero(U), zero(s)
    for j = 1:b:size(M,2)
        Ub, sb = ISVD.update!(Ub, sb, M[:,j:min(j+b-1, end)])
    end
    for (col1, col2) in zip(eachcol(U), eachcol(Ub))
        @test col1 ≈ col2 || col1 ≈ -col2
    end
    @test sb ≈ s

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
