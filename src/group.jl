function inverse_mat_sqrt(A::Symmetric; tol=1e-4)
    λ, ϕ = eigen(A)
    for i in eachindex(λ)
        λ[i] < tol && (λ[i] = tol)
    end
    return ϕ * Diagonal(1 ./ sqrt.(λ)) * ϕ'
end

function solve_Sb_equi(Σb::BlockDiagonal)
    Db = Matrix{eltype(Σ)}[]
    for Σbi in Σb.blocks
        push!(Db, inverse_mat_sqrt(Symmetric(Σbi)))
    end
    Db = BlockDiagonal(Db)
    λmin = Symmetric(Db * Σb * Db) |> eigvals |> minimum
    γb = min(1, 2λmin)
    Sb = BlockDiagonal(γb .* Σb.blocks)
    return Sb
end

function solve_s_group(Σ::BlockDiagonal, method=:group_equi)
    S = Matrix{eltype(Σ)}[]
    for Σb in Σ.blocks
        push!(S, solve_Sb_equi(Σb))
    end
    return BlockDiagonal(S)
end
