const _TURBO_ROTATION_TAIL_LENGTH = 10

_turbo_rotation_tol(::Type{T}) where T = 10eps(T)

function _negligible_rotation_tail(s, i::Int, idx_end::Int, tail_length::Int, ::Type{T}) where T
    if i ≥ idx_end && abs(s) ≤ _turbo_rotation_tol(T)
        tail_length += 1
    else
        tail_length = 0
    end
    return tail_length, tail_length ≥ _TURBO_ROTATION_TAIL_LENGTH
end

"""
    lowrankupdate_turbo!(C::Cholesky, v::AbstractVector)

Vectorized version of lowrankupdate!, source https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/LinearAlgebra/src/cholesky.jl#L707
Takes advantage of the fact that `v` is 0 everywhere except at 1 position
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

    tail_length = 0
    idx_start = findfirst(!iszero, v)
    isnothing(idx_start) && return C
    idx_end = something(findlast(!iszero, v))
    @inbounds for i = idx_start:n

        # Compute Givens rotation
        c, s, r = LinearAlgebra.givensAlgorithm(A[i,i], v[i])

        # The sparse-update tail is negligible once several consecutive
        # rotations are indistinguishable from the identity at machine scale.
        tail_length, should_stop = _negligible_rotation_tail(s, i, idx_end, tail_length, T)
        should_stop && break

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
    lowrankdowndate_turbo!(C::Cholesky, v::AbstractVector)

Vectorized version of lowrankdowndate!, source https://github.com/JuliaLang/julia/blob/742b9abb4dd4621b667ec5bb3434b8b3602f96fd/stdlib/LinearAlgebra/src/cholesky.jl#L753
Takes advantage of the fact that `v` is 0 everywhere except at 1 position
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

    tail_length = 0
    idx_start = findfirst(!iszero, v)
    isnothing(idx_start) && return C
    idx_end = something(findlast(!iszero, v))
    @inbounds for i = idx_start:n

        Aii = A[i,i]

        # Compute Givens rotation
        s = v[i] / Aii
        s2 = abs2(s)
        if s2 > 1
            throw(LinearAlgebra.PosDefException(i))
        end
        c = sqrt(1 - abs2(s))

        # The sparse-update tail is negligible once several consecutive
        # rotations are indistinguishable from the identity at machine scale.
        tail_length, should_stop = _negligible_rotation_tail(s, i, idx_end, tail_length, T)
        should_stop && break

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

