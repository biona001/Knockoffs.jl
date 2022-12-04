"""
    lowrankupdate_turbo!(C::Cholesky, v::AbstractVector, S::AbstractMatrix)

Vectorized version of `lowrankupdate!`, takes advantage of the fact that `v` is 0
everywhere except at 1 position. 

Source: https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/LinearAlgebra/src/cholesky.jl#L707
"""
function lowrankupdate_turbo!(C::Cholesky{T}, v::AbstractVector) where T <: AbstractFloat
    A = C.factors
    n = length(v)
    if size(C, 1) != n
        throw(DimensionMismatch("updating vector must fit size of factorization"))
    end
    # if C.uplo == 'U'
    #     conj!(v)
    # end

    early_term = 0
    idx_start = something(findfirst(!iszero, v))
    @inbounds for i = idx_start:n

        # Compute Givens rotation
        c, s, r = LinearAlgebra.givensAlgorithm(A[i,i], v[i])

        # check for early termination
        if abs(s) < 1e-15
            early_term += 1
            early_term > 10 && break
        else
            early_term = 0
        end

        # Store new diagonal element
        A[i,i] = r

        # Update remaining elements in row/column
        if C.uplo == 'U'
            @turbo for j = i + 1:n
                Aij = A[i,j]
                vj  = v[j]
                A[i,j]  =   c*Aij + s*vj
                v[j]    = -s*Aij + c*vj
            end
        else
            @turbo for j = i + 1:n
                Aji = A[j,i]
                vj  = v[j]
                A[j,i]  =   c*Aji + s*vj
                v[j]    = -s*Aji + c*vj
            end
        end
    end
    return C
end

"""
    lowrankdowndate_turbo!(C::Cholesky, v::AbstractVector, S::AbstractMatrix)

Vectorized version of `lowrankdowndate!`, takes advantage of the fact that `v`
is 0 everywhere except at 1 position. 

Source: https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/LinearAlgebra/src/cholesky.jl#L753
"""
function lowrankdowndate_turbo!(C::Cholesky{T}, v::AbstractVector) where T <: AbstractFloat
    A = C.factors
    n = length(v)
    if size(C, 1) != n
        throw(DimensionMismatch("updating vector must fit size of factorization"))
    end
    # if C.uplo == 'U'
    #     conj!(v)
    # end

    early_term = 0
    idx_start = something(findfirst(!iszero, v))
    @inbounds for i = idx_start:n

        Aii = A[i,i]

        # Compute Givens rotation
        s = v[i] / Aii
        s2 = abs2(s)
        if s2 > 1
            throw(LinearAlgebra.PosDefException(i))
        end
        c = sqrt(1 - abs2(s))

        # check for early termination
        if abs(s) < 1e-15
            early_term += 1
            early_term > 10 && break
        else
            early_term = 0
        end

        # Store new diagonal element
        A[i,i] = c*Aii

        # Update remaining elements in row/column
        if C.uplo == 'U'
            @turbo for j = i + 1:n
                vj = v[j]
                Aij = (A[i,j] - s*vj)/c
                A[i,j] = Aij
                v[j] = -s*Aij + c*vj
            end
        else
            @turbo for j = i + 1:n
                vj = v[j]
                Aji = (A[j,i] - s*vj)/c
                A[j,i] = Aji
                v[j] = -s*Aji + c*vj
            end
        end
    end
    return C
end

# for stable updates, i.e. when robust=true, used only in group knockoffs
lowrankupdate!(C::Cholesky, v::AbstractVector, S::AbstractMatrix) = LinearAlgebra.lowrankupdate!(C, v)
lowrankdowndate!(C::Cholesky, v::AbstractVector, S::AbstractMatrix) = LinearAlgebra.lowrankdowndate!(C, v)

"""
    lowrankupdate_turbo!(C::Cholesky{T}, v::AbstractVector, S::AbstractMatrix)

Safer version of lowrankupdate_turbo!(C, v), used for group knockoffs. This is needed
because sometimes numerical issues causes non-PSD error. In such case, try computing C
afresh using `S`. 
"""
function lowrankupdate_turbo!(C::Cholesky{T}, v::AbstractVector, S::AbstractMatrix) where T <: AbstractFloat
    C = try
        fdsa
        lowrankupdate_turbo!(C, v)
    catch
        cholesky(S)
    end
    return C
end

"""
    lowrankdowndate_turbo!(C::Cholesky{T}, v::AbstractVector, S::AbstractMatrix)

Safer version of lowrankdowndate_turbo!(C, v), used for group knockoffs. This is needed
because sometimes numerical issues causes non-PSD error. In such case, try computing C
afresh using `S`. 
"""
function lowrankdowndate_turbo!(C::Cholesky{T}, v::AbstractVector, S::AbstractMatrix) where T <: AbstractFloat
    C = try
        fdsa
        lowrankdowndate_turbo!(C, v)
    catch
        cholesky(S)
    end
    return C
end
