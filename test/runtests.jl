using Test, LinearAlgebra
import ISVD

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
