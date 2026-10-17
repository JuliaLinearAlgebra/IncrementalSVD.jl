@testset "owned latent-history transports" begin
    for T in (Float32,Float64), kind in (:signed,:zero,:rankone,:repeated)
        m,r=9,3
        X=kind==:zero ? zeros(T,m,11) :
            kind==:rankone ? T.(collect(1:m))*reshape(T.(sin.(1:11)),1,:) :
            kind==:repeated ? T.(Matrix(I,m,m)[:,mod1.(1:11,m)]) : T.(randn(m,11))
        U=zeros(T,m,r); s=zeros(T,r); H=zeros(T,r,0)
        tol=T==Float32 ? 2e-4 : 2e-12
        for rg in (1:3,4:6,7:9,10:11)
            A=X[:,rg]; U0=copy(U); s0=copy(s)
            c=ISVD.Cache(U,A)
            Ud,sd=ISVD.update!(copy(U),copy(s),A)
            _,_,tr=ISVD.update_transport!(U,s,A,c)
            @test U==Ud && s==sd
            H=hcat(tr.M*H,tr.h)
            @test H*H' ≈ Diagonal(s.^2) rtol=tol atol=tol
            @test all(isfinite,H)
            savedM=copy(tr.M); savedh=copy(tr.h)
            ISVD.update!(copy(U),copy(s),A,c)
            @test tr.M==savedM && tr.h==savedh
            @test !Base.mightalias(tr.M,c.K) && !Base.mightalias(tr.h,c.UtA)
        end
    end
    # With enough rank, signed full-rank data have no discarded information.
    X=randn(8,9); U=zeros(8,8); s=zeros(8); H=zeros(8,0)
    for k=1:9
        _,_,tr=ISVD.update_transport!(U,s,X[:,k:k])
        H=hcat(tr.M*H,tr.h)
    end
    @test U*H ≈ X atol=2e-12
    U=zeros(5,2); s=zeros(2); old=copy(U)
    @test_throws ArgumentError ISVD.update_transport!(U,s,fill(NaN,5,1))
    @test U==old && s==zeros(2)
    @test_throws DimensionMismatch ISVD.update_transport!(U,s,zeros(4,1))
end
