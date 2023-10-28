# Use multiple dispatch to create functions that are easier for R to call
# 1. Inputs that requires Symbols now accept Strings
# 2. Inputs that requires Symmetric matrix now takes Matrix. In exchange, one 
#    the extra boolean argument `isCovariance` will now differentiate whether
#    the input matrix is a Symmetric covariance matrix, or design matrix X. 

function hc_partition_groups(data::AbstractMatrix, isCovariance::Bool; 
    cutoff = 0.5, min_clusters = 1, linkage::String="complete", 
    force_contiguous=false)
    if isCovariance
        return hc_partition_groups(Symmetric(data); cutoff=cutoff, 
            min_clusters=min_clusters, linkage=linkage, 
            force_contiguous=force_contiguous)
    else
        return hc_partition_groups(data; cutoff=cutoff, min_clusters=min_clusters,
            linkage=linkage, force_contiguous=force_contiguous)
    end
end

function id_partition_groups(data::AbstractMatrix, isCovariance::Bool; 
    rss_target = 0.5, force_contiguous=false)
    if isCovariance
        return id_partition_groups(Symmetric(data); rss_target=rss_target,
            force_contiguous=force_contiguous)
    else
        return id_partition_groups(data; rss_target=rss_target,
            force_contiguous=force_contiguous)
    end
end

function solve_s(Σ::Matrix, method::Union{Symbol, String}; 
    m::Number=1, kwargs...)
    return solve_s(Symmetric(Σ), method; m=m, kwargs...)
end

function solve_s_group(Σ::Matrix, groups::Vector{Int},
    method::Union{Symbol, String}; m::Number=1, kwargs...)
    return solve_s_group(Symmetric(Σ), groups, method; m=m, kwargs...)
end

function solve_s_graphical_group(Σ::Matrix, groups::Vector{Int},
    group_reps::AbstractVector{Int}, method::Union{Symbol, String}; 
    m::Number=1, verbose::Bool=false, kwargs...)
    return solve_s_graphical_group(Symmetric(Σ), groups, group_reps, 
        method; m=m, verbose=verbose, kwargs...)
end
