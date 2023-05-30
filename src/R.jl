# Use multiple dispatch to create functions that are easier for R to call
# 1. Inputs that requires Symbols now accept Strings
# 2. Variable names with unicode characters get renamed to something
#    without unicodes, e.g. X̃ => Xko
# 3. Inputs that requires Symmetric matrix now takes Matrix


function modelX_gaussian_knockoffs(
    X::AbstractMatrix, 
    method::String;
    m::Number = 1, 
    kwargs...) # kwargs = extra arguments for solve_s
    return modelX_gaussian_knockoffs(X, Symbol(method); m=Int(m), kwargs...)
end
function modelX_gaussian_knockoffs(
    X::AbstractMatrix, 
    method::String, 
    μ::AbstractVector, 
    Σ::AbstractMatrix;
    m::Number = 1, 
    kwargs...) # kwargs = extra arguments for solve_s
    return modelX_gaussian_knockoffs(
        X, Symbol(method), μ, Σ; m=Int(m), kwargs...
    )
end

function modelX_gaussian_group_knockoffs(
    X::AbstractMatrix, 
    method::String, 
    groups::AbstractVector;
    m::Number = 1, 
    kwargs...) # kwargs = extra arguments for solve_s_group
    return modelX_gaussian_group_knockoffs(X, Symbol(method), Int.(groups); 
        m=Int(m), kwargs...
    )
end
function modelX_gaussian_group_knockoffs(
    X::AbstractMatrix, 
    method::String, 
    groups::AbstractVector, 
    μ::AbstractVector, 
    Σ::AbstractMatrix;
    m::Number = 1, 
    kwargs...) # kwargs = extra arguments for solve_s_group
    return modelX_gaussian_group_knockoffs(
        X, Symbol(method), Int.(groups), μ, Σ; 
        m=Int(m), kwargs...
    )
end

# defines ko.f for f = Xko (does not help R users however)
# function Base.getproperty(ko::Knockoff, f::Symbol)
#     if f == :Xko
#         return ko.X̃
#     else # fallback to getfield
#         return getfield(ko, f)
#     end
# end
