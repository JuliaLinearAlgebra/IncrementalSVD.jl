using Test, LinearAlgebra
import ISVD

@testset "ISVD" begin
    r = 5
    A = rand(20, r)
    B = rand(r, 30)
    M = A*B
    U = Array{Float64}(undef, size(A,1), 0)
    s = Array{Float64}(undef, 0)
    for j = 1:r:size(M,2)
        U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
    end
    Un = sn = nothing
    for j = 1:r:size(M,2)
        Un, sn = ISVD.update_U_s!(Un, sn, M[:,j:j+r-1])
    end
    @test sn ≈ s && Un ≈ U

    # Test that A is within the span of U
    pA = U*(U'*A)
    @test pA ≈ A

    for j = 1:r:size(M,2)
        U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
    end
    U1, s1, _ = svd(hcat(M, M))
    @test s ≈ s1[1:r]
    @test abs.(U1[:,1:r]'*U) ≈ Matrix{Float64}(I,r,r)

    # Mismatched element types
    U = Array{Float32}(undef, size(A,1), 0)
    s = Array{Float32}(undef, 0)
    for j = 1:r:size(M,2)
        U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
    end
    @test eltype(U) == Float32

    # NaN imputation
    M[3,8] = NaN  # imputation "works" starting with the 2nd block
    U = Array{Float64}(undef, size(A,1), 0)
    s = Array{Float64}(undef, 0)
    for j = 1:r:size(M,2)
        U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
    end
    @test !any(isnan, U)
    @test !any(isnan, s)
    pA = U*(U'*A)
    @test pA ≈ A

    # Test that first-block NaNs don't break things catastrophically
    M[5,2] = NaN
    U = Array{Float64}(undef, size(A,1), 0)
    s = Array{Float64}(undef, 0)
    for j = 1:r:size(M,2)
        U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
    end
    @test !any(isnan, U)
    @test !any(isnan, s)
end
