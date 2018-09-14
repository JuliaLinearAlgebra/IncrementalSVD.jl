using Base.Test
import CellSegmentation, ISVD

r = 5
A = rand(20, r)
B = rand(r, 30)
M = A*B
U = Array{Float64}(size(A,1), 0)
s = Array{Float64}(0)
for j = 1:r:size(M,2)
    U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
end

# Test that A is within the span of U
pA = U*(U'*A)
@test pA ≈ A

for j = 1:r:size(M,2)
    U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
end
U1, s1, _ = svd(hcat(M, M))
@test s ≈ s1[1:r]
@test abs.(U1[:,1:r]'*U) ≈ eye(r,r)

# Mismatched element types
U = Array{Float32}(size(A,1), 0)
s = Array{Float32}(0)
for j = 1:r:size(M,2)
    U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
end
@test eltype(U) == Float32

# NaN imputation
M[3,8] = NaN  # imputation "works" starting with the 2nd block
U = Array{Float64}(size(A,1), 0)
s = Array{Float64}(0)
for j = 1:r:size(M,2)
    U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
end
@test !any(isnan, U)
@test !any(isnan, s)
pA = U*(U'*A)
@test pA ≈ A

# Test that first-block NaNs don't break things catastrophically
M[5,2] = NaN
U = Array{Float64}(size(A,1), 0)
s = Array{Float64}(0)
for j = 1:r:size(M,2)
    U, s = ISVD.update_U_s!(U, s, M[:,j:j+r-1])
end
@test !any(isnan, U)
@test !any(isnan, s)
