"""
    add_two(x::T) where {T<:Number} 

Adds two to the input number.
"""
function add_two(x::T) where {T<:Number}
    x + T(2) 
end

"""
    add_two(x::AbstractArray{T,N}) where {T,N} 

Adds two to an array element-wise.
"""
function add_two(x::AbstractArray{T,N}) where {T,N}
    x .+ T(2)
end