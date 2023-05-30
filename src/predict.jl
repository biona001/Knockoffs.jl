function predict(
    model::LassoKnockoffFilter{T},
    xtest::AbstractMatrix{T}
    ) where T
    ŷs = Vector{T}[]
    d = model.d
    link = canonicallink(d)
    for i in 1:length(model.betas)
        # compute mean: η = a0 .+ Xβ̂
        η = fill(model.a0[i], size(xtest, 1))
        BLAS.gemv!('N', one(T), xtest, model.betas[i], one(T), η)
        # apply inverse link to get mean
        μ = GLM.linkinv.(link, η)
        push!(ŷs, μ)
    end
    return ŷs
end

function R2(ŷ::AbstractVector, y::AbstractVector)
    tss = y .- mean(y)
    rss = y .- ŷ
    return 1 - dot(rss, rss) / dot(tss, tss)
end

# auc functions from https://github.com/milanflach/MultivariateAnomalies.jl/blob/master/src/AUC.jl
function auc(scores, events; increasing::Bool = true)
    s = sortperm(reshape(scores,length(scores)),rev=increasing);
    length(scores) == length(events) || error("Scores and events must have same number of elements")
    f=scores[s]
    L=events[s]
    fp=0
    tp=0
    fpprev=0
    tpprev=0
    A=0.0
    fprev=-Inf
    P=sum(L)
    N=length(L)-P
    for i=1:length(L)
        if f[i]!=fprev
            A+=trap_area(fp,fpprev,tp,tpprev)
            @inbounds fprev=f[i]
            fpprev=fp
            tpprev=tp
        end
        if isextreme(L[i])
            tp+=1
        else
            fp+=1
        end
    end
    A+=trap_area(N,fpprev,P,tpprev)
    A=A/(P*N)
end
function trap_area(x1,x2,y1,y2)
    b=abs(x1-x2)
    h=0.5*(y1+y2)
    return b*h
end
isextreme(l::Bool)=l
isextreme(l::Integer)=l>0
