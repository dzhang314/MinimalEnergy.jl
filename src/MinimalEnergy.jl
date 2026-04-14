module MinimalEnergy

using KernelAbstractions
using MultiFloats: div_r, inv_r, rsqrt_r, sqrt_r
using Random: AbstractRNG, default_rng, rand!, randn

################################################################################


function common_backend(array, arrays...)
    backend = get_backend(array)
    for a in arrays
        @assert backend == get_backend(a)
    end
    return backend
end


@inline function butterfly_sum!(
    values::AbstractArray{T,1},
    block_size::Int,
    local_index::Int,
) where {T}
    @inbounds begin
        stride = block_size >> 1
        while stride > 0
            if local_index <= stride
                values[local_index] += values[local_index+stride]
            end
            @synchronize
            stride >>= 1
        end
        result = values[1]
        @synchronize
        return result
    end
end


@kernel inbounds = true function particle_sum_kernel!(
    sums::AbstractArray{T,1},
    alpha::T,
    values::AbstractArray{T,2},
    num_particles::Int,
) where {T}
    block_size = @uniform(@groupsize()[1])
    temp = @localmem(T, (block_size,))

    _, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)

    acc = zero(T)
    @inbounds for i = local_index:block_size:num_particles
        acc += values[i, lane]
    end
    temp[local_index] = acc
    @synchronize
    result = butterfly_sum!(temp, block_size, local_index)
    if isone(local_index)
        sums[lane] = alpha * result
    end
end


function particle_sum!(
    sums::AbstractArray{T,1},
    alpha::T,
    values::AbstractArray{T,2},
    num_particles::Int,
    num_lanes::Int,
    block_size::Int,
) where {T}
    @assert ispow2(block_size)
    backend = common_backend(sums, values)
    kernel = particle_sum_kernel!(backend, (block_size, 1))
    kernel(sums, alpha, values, num_particles, ndrange=(block_size, num_lanes))
    return sums
end


@kernel inbounds = true function batched_dot_product_kernel!(
    particle_products::AbstractArray{T,2},
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    num_particles::Int,
) where {T}
    block_size = @uniform(@groupsize()[1])
    block_index, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)
    i = (block_index - 1) * block_size + local_index

    if i <= num_particles
        particle_products[i, lane] = (
            u[1, i, lane] * v[1, i, lane] +
            u[2, i, lane] * v[2, i, lane] +
            u[3, i, lane] * v[3, i, lane])
    end
end


function batched_dot_product!(
    lane_products::AbstractArray{T,1},
    particle_products::AbstractArray{T,2},
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    num_particles::Int,
    num_lanes::Int,
    block_size::Int,
    sum_block_size::Int,
) where {T}
    backend = common_backend(lane_products, particle_products, u, v)
    kernel = batched_dot_product_kernel!(backend, (block_size, 1))
    num_particles_up = cld(num_particles, block_size) * block_size
    kernel(particle_products, u, v, num_particles,
        ndrange=(num_particles_up, num_lanes))
    particle_sum!(lane_products, one(T), particle_products,
        num_particles, num_lanes, sum_block_size)
    return (lane_products, particle_products)
end


################################################################################


export coulomb_energy_force!


@kernel inbounds = true function coulomb_energy_force_kernel!(
    particle_energies::AbstractArray{T,2},
    F::AbstractArray{T,3},
    x::AbstractArray{T,3},
    num_particles::Int,
) where {T}
    block_size = @uniform(@groupsize()[1])
    block_x = @localmem(T, (3, block_size))

    block_index, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)
    i = (block_index - 1) * block_size + local_index
    active_i = (i <= num_particles)

    xi = active_i ? x[1, i, lane] : zero(T)
    yi = active_i ? x[2, i, lane] : zero(T)
    zi = active_i ? x[3, i, lane] : zero(T)
    energy = zero(T)
    force_x = zero(T)
    force_y = zero(T)
    force_z = zero(T)
    for block_offset = 0:block_size:(num_particles-1)
        j = block_offset + local_index
        active_j = (j <= num_particles)
        block_x[1, local_index] = active_j ? x[1, j, lane] : zero(T)
        block_x[2, local_index] = active_j ? x[2, j, lane] : zero(T)
        block_x[3, local_index] = active_j ? x[3, j, lane] : zero(T)
        @synchronize
        if active_i
            active_block_size = min(block_size, num_particles - block_offset)
            for k = 1:active_block_size
                if k + block_offset != i
                    dx = xi - block_x[1, k]
                    dy = yi - block_x[2, k]
                    dz = zi - block_x[3, k]
                    inv_r = rsqrt_r(abs2(dx) + abs2(dy) + abs2(dz))
                    inv_r2 = inv_r * inv_r
                    inv_r3 = inv_r * inv_r2
                    energy += inv_r
                    force_x += inv_r3 * dx
                    force_y += inv_r3 * dy
                    force_z += inv_r3 * dz
                end
            end
        end
        @synchronize
    end
    if active_i
        particle_energies[i, lane] = energy
        F[1, i, lane] = force_x
        F[2, i, lane] = force_y
        F[3, i, lane] = force_z
    end
end


function coulomb_energy_force!(
    lane_energies::AbstractArray{T,1},
    particle_energies::AbstractArray{T,2},
    F::AbstractArray{T,3},
    x::AbstractArray{T,3},
    num_particles::Int,
    num_lanes::Int,
    block_size::Int,
    sum_block_size::Int,
) where {T}
    _half = inv_r(one(T) + one(T))
    backend = common_backend(lane_energies, particle_energies, F, x)
    kernel = coulomb_energy_force_kernel!(backend, (block_size, 1))
    num_particles_up = cld(num_particles, block_size) * block_size
    kernel(particle_energies, F, x, num_particles,
        ndrange=(num_particles_up, num_lanes))
    particle_sum!(lane_energies, _half, particle_energies,
        num_particles, num_lanes, sum_block_size)
    return (lane_energies, particle_energies, F)
end


################################################################################


export sphere_step!, sphere_tangent!, sphere_directional_derivative!


@kernel inbounds = true function sphere_step_kernel!(
    x::AbstractArray{T,3},
    step_sizes::AbstractArray{T,1},
    x0::AbstractArray{T,3},
    u0::AbstractArray{T,3},
    num_particles::Int,
) where {T}
    block_size = @uniform(@groupsize()[1])
    block_index, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)
    i = (block_index - 1) * block_size + local_index

    if i <= num_particles
        step_size = step_sizes[lane]
        xi = x0[1, i, lane] + step_size * u0[1, i, lane]
        yi = x0[2, i, lane] + step_size * u0[2, i, lane]
        zi = x0[3, i, lane] + step_size * u0[3, i, lane]
        inv_norm = rsqrt_r(abs2(xi) + abs2(yi) + abs2(zi))
        x[1, i, lane] = inv_norm * xi
        x[2, i, lane] = inv_norm * yi
        x[3, i, lane] = inv_norm * zi
    end
end


function sphere_step!(
    x::AbstractArray{T,3},
    step_sizes::AbstractArray{T,1},
    x0::AbstractArray{T,3},
    u0::AbstractArray{T,3},
    num_particles::Int,
    num_lanes::Int,
    block_size::Int,
) where {T}
    backend = common_backend(x, step_sizes, x0, u0)
    kernel = sphere_step_kernel!(backend, (block_size, 1))
    num_particles_up = cld(num_particles, block_size) * block_size
    kernel(x, step_sizes, x0, u0, num_particles,
        ndrange=(num_particles_up, num_lanes))
    return x
end


@kernel inbounds = true function sphere_tangent_kernel!(
    d::AbstractArray{T,3},
    x::AbstractArray{T,3},
    num_particles::Int,
) where {T}
    block_size = @uniform(@groupsize()[1])
    block_index, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)
    i = (block_index - 1) * block_size + local_index

    if i <= num_particles
        xi = x[1, i, lane]
        yi = x[2, i, lane]
        zi = x[3, i, lane]
        overlap = d[1, i, lane] * xi + d[2, i, lane] * yi + d[3, i, lane] * zi
        d[1, i, lane] -= overlap * xi
        d[2, i, lane] -= overlap * yi
        d[3, i, lane] -= overlap * zi
    end
end


function sphere_tangent!(
    d::AbstractArray{T,3},
    x::AbstractArray{T,3},
    num_particles::Int,
    num_lanes::Int,
    block_size::Int,
) where {T}
    backend = common_backend(d, x)
    kernel = sphere_tangent_kernel!(backend, (block_size, 1))
    num_particles_up = cld(num_particles, block_size) * block_size
    kernel(d, x, num_particles, ndrange=(num_particles_up, num_lanes))
    return d
end


@kernel inbounds = true function sphere_directional_derivative_kernel!(
    particle_derivatives::AbstractArray{T,2},
    x::AbstractArray{T,3},
    F::AbstractArray{T,3},
    x0::AbstractArray{T,3},
    u0::AbstractArray{T,3},
    num_particles::Int,
) where {T}
    block_size = @uniform(@groupsize()[1])
    block_index, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)
    i = (block_index - 1) * block_size + local_index

    if i <= num_particles
        xi = x[1, i, lane]
        yi = x[2, i, lane]
        zi = x[3, i, lane]
        x0i = x0[1, i, lane]
        y0i = x0[2, i, lane]
        z0i = x0[3, i, lane]
        u0i = u0[1, i, lane]
        v0i = u0[2, i, lane]
        w0i = u0[3, i, lane]
        cos_angle = x0i * xi + y0i * yi + z0i * zi
        overlap = u0i * xi + v0i * yi + w0i * zi
        dxi = cos_angle * (u0i - overlap * xi)
        dyi = cos_angle * (v0i - overlap * yi)
        dzi = cos_angle * (w0i - overlap * zi)
        particle_derivatives[i, lane] =
            F[1, i, lane] * dxi + F[2, i, lane] * dyi + F[3, i, lane] * dzi
    end
end


function sphere_directional_derivative!(
    lane_derivatives::AbstractArray{T,1},
    particle_derivatives::AbstractArray{T,2},
    x::AbstractArray{T,3},
    F::AbstractArray{T,3},
    x0::AbstractArray{T,3},
    u0::AbstractArray{T,3},
    num_particles::Int,
    num_lanes::Int,
    block_size::Int,
    sum_block_size::Int,
) where {T}
    backend = common_backend(lane_derivatives, particle_derivatives,
        x, F, x0, u0)
    kernel = sphere_directional_derivative_kernel!(backend, (block_size, 1))
    num_particles_up = cld(num_particles, block_size) * block_size
    kernel(particle_derivatives, x, F, x0, u0, num_particles,
        ndrange=(num_particles_up, num_lanes))
    particle_sum!(lane_derivatives, one(T), particle_derivatives,
        num_particles, num_lanes, sum_block_size)
    return (lane_derivatives, particle_derivatives)
end


################################################################################


export lbfgs!


@inline function scale_3d!(
    d::AbstractArray{T,3},
    alpha::T,
    index_range::AbstractRange{Int},
    lane::Int,
) where {T}
    @inbounds for i in index_range
        d[1, i, lane] *= alpha
        d[2, i, lane] *= alpha
        d[3, i, lane] *= alpha
    end
    return nothing
end


@inline function axpy_3d!(
    dst::AbstractArray{T,3},
    alpha::T,
    src::AbstractArray{T,4},
    index_range::AbstractRange{Int},
    lane::Int,
    h::Int,
) where {T}
    @inbounds for i in index_range
        dst[1, i, lane] += alpha * src[1, i, lane, h]
        dst[2, i, lane] += alpha * src[2, i, lane, h]
        dst[3, i, lane] += alpha * src[3, i, lane, h]
    end
    return nothing
end


@inline function dot_3d!(
    temp::AbstractArray{T,1},
    u::AbstractArray{T,3},
    v::AbstractArray{T,4},
    num_particles::Int,
    lane::Int,
    h::Int,
    block_size::Int,
    local_index::Int,
) where {T}
    @inbounds begin
        acc = zero(T)
        for i = local_index:block_size:num_particles
            acc += (u[1, i, lane] * v[1, i, lane, h] +
                    u[2, i, lane] * v[2, i, lane, h] +
                    u[3, i, lane] * v[3, i, lane, h])
        end
        temp[local_index] = acc
        @synchronize
        return butterfly_sum!(temp, block_size, local_index)
    end
end


@kernel inbounds = true function lbfgs_kernel!(
    d::AbstractArray{T,3},
    x::AbstractArray{T,3},
    F::AbstractArray{T,3},
    x0::AbstractArray{T,3},
    F0::AbstractArray{T,3},
    s::AbstractArray{T,4},
    y::AbstractArray{T,4},
    rho::AbstractArray{T,2},
    num_particles::Int,
    h::NTuple{4,Int},
) where {T}
    block_size = @uniform(@groupsize()[1])
    temp1 = @localmem(T, (block_size,))
    temp2 = @localmem(T, (block_size,))
    temp3 = @localmem(T, (block_size,))

    _, lane = @index(Group, NTuple)
    local_index, _ = @index(Local, NTuple)

    acc_sf = zero(T)
    acc_sy = zero(T)
    acc_yy = zero(T)
    for i = local_index:block_size:num_particles
        # Initialize step direction to force (steepest descent) direction.
        Fx = F[1, i, lane]
        Fy = F[2, i, lane]
        Fz = F[3, i, lane]
        d[1, i, lane] = Fx
        d[2, i, lane] = Fy
        d[3, i, lane] = Fz
        # Compute s0 := change in position since previous step.
        sx = x[1, i, lane] - x0[1, i, lane]
        sy = x[2, i, lane] - x0[2, i, lane]
        sz = x[3, i, lane] - x0[3, i, lane]
        s[1, i, lane, h[1]] = sx
        s[2, i, lane, h[1]] = sy
        s[3, i, lane, h[1]] = sz
        # Compute y0 := change in force since previous step.
        yx = Fx - F0[1, i, lane]
        yy = Fy - F0[2, i, lane]
        yz = Fz - F0[3, i, lane]
        y[1, i, lane, h[1]] = yx
        y[2, i, lane, h[1]] = yy
        y[3, i, lane, h[1]] = yz
        # Compute (s0 dot f), (s0 dot y0), and (y0 dot y0).
        acc_sf += sx * Fx + sy * Fy + sz * Fz
        acc_sy += sx * yx + sy * yy + sz * yz
        acc_yy += yx * yx + yy * yy + yz * yz
    end
    temp1[local_index] = acc_sf
    temp2[local_index] = acc_sy
    temp3[local_index] = acc_yy
    @synchronize
    sf = butterfly_sum!(temp1, block_size, local_index)
    sy = butterfly_sum!(temp2, block_size, local_index)
    yy = butterfly_sum!(temp3, block_size, local_index)

    gamma = div_r(-sy, yy)
    rho1 = inv_r(sy)
    if !(isfinite(rho1) & isfinite(gamma) & signbit(rho1) & !signbit(gamma))
        gamma = one(T)
        rho1 = zero(T)
    end
    if isone(local_index)
        rho[lane, h[1]] = rho1
    end
    rho2 = rho[lane, h[2]]
    rho3 = rho[lane, h[3]]
    rho4 = rho[lane, h[4]]

    alpha1 = rho1 * sf
    axpy_3d!(d, -alpha1, y, local_index:block_size:num_particles, lane, h[1])

    alpha2 = rho2 * dot_3d!(temp1, d, s,
        num_particles, lane, h[2], block_size, local_index)
    axpy_3d!(d, -alpha2, y, local_index:block_size:num_particles, lane, h[2])

    alpha3 = rho3 * dot_3d!(temp1, d, s,
        num_particles, lane, h[3], block_size, local_index)
    axpy_3d!(d, -alpha3, y, local_index:block_size:num_particles, lane, h[3])

    alpha4 = rho4 * dot_3d!(temp1, d, s,
        num_particles, lane, h[4], block_size, local_index)
    axpy_3d!(d, -alpha4, y, local_index:block_size:num_particles, lane, h[4])
    scale_3d!(d, gamma, local_index:block_size:num_particles, lane)

    c4 = -(alpha4 + rho4 * dot_3d!(temp1, d, y,
        num_particles, lane, h[4], block_size, local_index))
    axpy_3d!(d, c4, s, local_index:block_size:num_particles, lane, h[4])

    c3 = -(alpha3 + rho3 * dot_3d!(temp1, d, y,
        num_particles, lane, h[3], block_size, local_index))
    axpy_3d!(d, c3, s, local_index:block_size:num_particles, lane, h[3])

    c2 = -(alpha2 + rho2 * dot_3d!(temp1, d, y,
        num_particles, lane, h[2], block_size, local_index))
    axpy_3d!(d, c2, s, local_index:block_size:num_particles, lane, h[2])

    c1 = -(alpha1 + rho1 * dot_3d!(temp1, d, y,
        num_particles, lane, h[1], block_size, local_index))
    axpy_3d!(d, c1, s, local_index:block_size:num_particles, lane, h[1])
end


function lbfgs!(
    d::AbstractArray{T,3},
    s::AbstractArray{T,4},
    y::AbstractArray{T,4},
    rho::AbstractArray{T,2},
    x::AbstractArray{T,3},
    F::AbstractArray{T,3},
    x0::AbstractArray{T,3},
    F0::AbstractArray{T,3},
    num_particles::Int,
    num_lanes::Int,
    h::NTuple{4,Int},
    block_size::Int,
) where {T}
    @assert ispow2(block_size)
    backend = common_backend(d, x, F, x0, F0, s, y, rho)
    kernel = lbfgs_kernel!(backend, (block_size, 1))
    kernel(d, x, F, x0, F0, s, y, rho, num_particles, h,
        ndrange=(block_size, num_lanes))
    return d
end


################################################################################


export BlockSizeConfiguration, ThomsonOptimizer,
    initialize!, evaluate_step!, accept_step!


struct BlockSizeConfiguration
    dot::Int
    dot_sum::Int
    coulomb::Int
    coulomb_sum::Int
    step::Int
    tangent::Int
    derivative::Int
    derivative_sum::Int
    lbfgs::Int
end


struct ThomsonOptimizer{T,
    A1<:AbstractArray{T,1},A2<:AbstractArray{T,2},
    A3<:AbstractArray{T,3},A4<:AbstractArray{T,4}}

    num_particles::Int
    num_lanes::Int
    block_sizes::BlockSizeConfiguration
    num_iterations::Array{Int,0}

    # Current point
    E_host::Vector{T}
    dE_host::Vector{T}
    x::A3
    F::A3

    # Previous point
    E0_host::Vector{T}
    dE0_host::Vector{T}
    x0::A3
    F0::A3

    # Line search data
    optimization_done_host::BitVector
    line_search_done_host::BitVector
    bracketed_host::BitVector
    modified_host::BitVector
    step_sizes_host::Vector{T}
    w_host::Vector{T}
    w0_host::Vector{T}
    a_host::Vector{T}
    Ea_host::Vector{T}
    dEa_host::Vector{T}
    b_host::Vector{T}
    Eb_host::Vector{T}
    dEb_host::Vector{T}

    # L-BFGS data
    d::A3
    s::A4
    y::A4
    rho::A2

    # Workspace arrays
    temp_particle::A2
    temp_lane::A1
end


@inline function copyto_negate!(
    dst::AbstractArray{T,1},
    src::AbstractArray{T,1},
) where {T}
    copyto!(dst, src)
    @inbounds for i in eachindex(dst)
        dst[i] = -dst[i]
    end
    return dst
end


@inline poison!(a::AbstractArray{<:Integer}) = rand!(a)
@inline poison!(a::AbstractArray{T}) where {T} = fill!(a, T(NaN))
@inline poison!(a::Tuple) = poison!.(a)


function initialize!(opt::ThomsonOptimizer{T}) where {T}

    poison!((opt.E_host, opt.dE_host, opt.F,
        opt.E0_host, opt.dE0_host, opt.x0, opt.F0,
        opt.optimization_done_host, opt.line_search_done_host,
        opt.bracketed_host, opt.modified_host, opt.step_sizes_host,
        opt.w_host,
        opt.w0_host,
        opt.a_host, opt.Ea_host, opt.dEa_host,
        opt.b_host, opt.Eb_host, opt.dEb_host,
        opt.d, opt.s, opt.y, opt.rho,
        opt.temp_particle, opt.temp_lane))

    # Take step with size zero to normalize points.
    fill!(opt.d, zero(T))
    fill!(opt.temp_lane, zero(T))
    sphere_step!(opt.x, opt.temp_lane, opt.x, opt.d,
        opt.num_particles, opt.num_lanes, opt.block_sizes.step)

    # Initialize energy and forces.
    coulomb_energy_force!(opt.temp_lane, opt.temp_particle, opt.F,
        opt.x, opt.num_particles, opt.num_lanes,
        opt.block_sizes.coulomb, opt.block_sizes.coulomb_sum)
    copyto!(opt.E_host, opt.temp_lane)
    sphere_tangent!(opt.F,
        opt.x, opt.num_particles, opt.num_lanes, opt.block_sizes.tangent)

    # Set initial step direction to force direction (steepest descent).
    copyto!(opt.d, opt.F)
    batched_dot_product!(opt.temp_lane, opt.temp_particle,
        opt.d, opt.F, opt.num_particles, opt.num_lanes,
        opt.block_sizes.dot, opt.block_sizes.dot_sum)
    copyto_negate!(opt.dE_host, opt.temp_lane)

    # Copy initial point to previous point.
    copyto!(opt.E0_host, opt.E_host)
    copyto!(opt.dE0_host, opt.dE_host)
    copyto!(opt.x0, opt.x)
    copyto!(opt.F0, opt.F)

    # Activate all lanes.
    fill!(opt.optimization_done_host, false)
    fill!(opt.line_search_done_host, false)

    # Zero out L-BFGS buffers.
    fill!(opt.s, zero(T))
    fill!(opt.y, zero(T))
    fill!(opt.rho, zero(T))

    return opt
end


function ThomsonOptimizer(
    backend::Backend,
    ::Type{T},
    num_particles::Int,
    num_lanes::Int;
    block_sizes::BlockSizeConfiguration=BlockSizeConfiguration(
        128, 128, 128, 128, 128, 128, 128, 128, 128),
    rng::AbstractRNG=default_rng(),
) where {T}
    n = num_particles
    k = num_lanes
    x_host = Array{T,3}(undef, 3, n, k)
    for lane = 1:k, i = 1:n
        x_temp = randn(rng, T)
        y_temp = randn(rng, T)
        z_temp = randn(rng, T)
        inv_norm = rsqrt_r(abs2(x_temp) + abs2(y_temp) + abs2(z_temp))
        x_host[1, i, lane] = inv_norm * x_temp
        x_host[2, i, lane] = inv_norm * y_temp
        x_host[3, i, lane] = inv_norm * z_temp
    end
    x = allocate(backend, T, 3, n, k)
    copyto!(x, x_host)
    return initialize!(ThomsonOptimizer(n, k, block_sizes, fill(0),
        Vector{T}(undef, k), Vector{T}(undef, k),
        x, allocate(backend, T, 3, n, k),
        Vector{T}(undef, k), Vector{T}(undef, k),
        allocate(backend, T, 3, n, k), allocate(backend, T, 3, n, k),
        BitVector(undef, k), BitVector(undef, k),
        BitVector(undef, k), BitVector(undef, k), Vector{T}(undef, k),
        Vector{T}(undef, k), Vector{T}(undef, k),
        Vector{T}(undef, k), Vector{T}(undef, k), Vector{T}(undef, k),
        Vector{T}(undef, k), Vector{T}(undef, k), Vector{T}(undef, k),
        allocate(backend, T, 3, n, k),
        allocate(backend, T, 3, n, k, 4),
        allocate(backend, T, 3, n, k, 4),
        allocate(backend, T, k, 4),
        allocate(backend, T, n, k), allocate(backend, T, k)))
end


function evaluate_step!(opt::ThomsonOptimizer)
    copyto!(opt.temp_lane, opt.step_sizes_host)
    sphere_step!(opt.x, opt.temp_lane, opt.x0, opt.d,
        opt.num_particles, opt.num_lanes, opt.block_sizes.step)
    coulomb_energy_force!(opt.temp_lane, opt.temp_particle, opt.F,
        opt.x, opt.num_particles, opt.num_lanes,
        opt.block_sizes.coulomb, opt.block_sizes.coulomb_sum)
    copyto!(opt.E_host, opt.temp_lane)
    sphere_tangent!(opt.F,
        opt.x, opt.num_particles, opt.num_lanes, opt.block_sizes.tangent)
    sphere_directional_derivative!(opt.temp_lane, opt.temp_particle,
        opt.x, opt.F, opt.x0, opt.d, opt.num_particles, opt.num_lanes,
        opt.block_sizes.derivative, opt.block_sizes.derivative_sum)
    copyto_negate!(opt.dE_host, opt.temp_lane)
    return opt
end


function accept_step!(opt::ThomsonOptimizer)
    copyto!(opt.temp_lane, opt.step_sizes_host)
    sphere_step!(opt.x, opt.temp_lane, opt.x0, opt.d,
        opt.num_particles, opt.num_lanes, opt.block_sizes.step)
    coulomb_energy_force!(opt.temp_lane, opt.temp_particle, opt.F,
        opt.x, opt.num_particles, opt.num_lanes,
        opt.block_sizes.coulomb, opt.block_sizes.coulomb_sum)
    copyto!(opt.E_host, opt.temp_lane)
    sphere_tangent!(opt.F,
        opt.x, opt.num_particles, opt.num_lanes, opt.block_sizes.tangent)
    return opt
end


function lbfgs!(opt::ThomsonOptimizer{T}) where {T}
    h0 = (opt.num_iterations[] & 3) + 1
    h1 = ((opt.num_iterations[] - 1) & 3) + 1
    h2 = ((opt.num_iterations[] - 2) & 3) + 1
    h3 = ((opt.num_iterations[] - 3) & 3) + 1
    lbfgs!(opt.d, opt.s, opt.y, opt.rho,
        opt.x, opt.F, opt.x0, opt.F0, opt.num_particles, opt.num_lanes,
        (h0, h1, h2, h3), opt.block_sizes.lbfgs)
    sphere_tangent!(opt.d,
        opt.x, opt.num_particles, opt.num_lanes, opt.block_sizes.tangent)
    batched_dot_product!(opt.temp_lane, opt.temp_particle,
        opt.d, opt.F, opt.num_particles, opt.num_lanes,
        opt.block_sizes.dot, opt.block_sizes.dot_sum)
    copyto_negate!(opt.dE_host, opt.temp_lane)
    return opt
end


function reset_lbfgs_history!(opt::ThomsonOptimizer{T}, lane::Int) where {T}
    for h = 1:4
        fill!(view(opt.s, :, :, lane, h), zero(T))
        fill!(view(opt.y, :, :, lane, h), zero(T))
        fill!(view(opt.rho, lane:lane, h:h), zero(T))
    end
    copyto!(view(opt.d, :, :, lane), view(opt.F, :, :, lane))
    return opt
end


################################################################################


export LineSearchParameters, line_search!


struct LineSearchParameters{T}
    c1::T
    c2::T
    contraction_factor::T
    extrapolation_lower_bound::T
    extrapolation_upper_bound::T
end


@inline function secant_minimizer(
    a::T, ga::T,
    b::T, gb::T,
) where {T}
    delta_x = b - a
    delta_g = gb - ga
    return a - div_r(ga * delta_x, delta_g)
end


@inline function quadratic_minimizer(
    a::T, fa::T, ga::T,
    b::T, fb::T,
) where {T}
    delta_x = b - a
    delta_f = div_r(fb - fa, delta_x)
    half_delta_g = delta_f - ga
    delta_g = half_delta_g + half_delta_g
    return a - div_r(ga * delta_x, delta_g)
end


@inline function cubic_minimizer(
    a::T, fa::T, ga::T,
    b::T, fb::T, gb::T,
) where {T}
    delta_x = b - a
    delta_f = div_r(fb - fa, delta_x)
    two_delta_f = delta_f + delta_f
    three_delta_f = two_delta_f + delta_f
    gc = three_delta_f - (ga + gb)
    s = max(abs(ga), abs(gb), abs(gc))
    inv_s = inv_r(s)
    sqrt_disc = copysign(s, delta_x) * sqrt_r(max(zero(T),
        abs2(inv_s * gc) - (inv_s * ga) * (inv_s * gb)))
    temp = sqrt_disc - ga
    return a + div_r(temp - gc, (temp + sqrt_disc) + gb) * delta_x
end


@inline function higher_value_step_size(
    a::T, fa::T, ga::T,
    t::T, ft::T, gt::T,
) where {T}
    _half = inv_r(one(T) + one(T))
    tq = quadratic_minimizer(a, fa, ga, t, ft)
    tc = cubic_minimizer(a, fa, ga, t, ft, gt)
    return (abs(tc - a) < abs(tq - a)) ? tc : (_half * (tq + tc))
end


@inline function opposite_slope_step_size(
    a::T, fa::T, ga::T,
    t::T, ft::T, gt::T,
) where {T}
    ts = secant_minimizer(a, ga, t, gt)
    tc = cubic_minimizer(a, fa, ga, t, ft, gt)
    return (abs(tc - t) > abs(ts - t)) ? tc : ts
end


@inline function reduced_slope_step_size(
    a::T, fa::T, ga::T,
    b::T,
    t::T, ft::T, gt::T,
    t_min::T, t_max::T, bracketed::Bool, contraction_factor::T,
) where {T}
    delta_x = t - a
    delta_f = div_r(ft - fa, delta_x)
    two_delta_f = delta_f + delta_f
    three_delta_f = two_delta_f + delta_f
    gc = three_delta_f - (ga + gt)
    s = max(abs(ga), abs(gt), abs(gc))
    inv_s = inv_r(s)
    sqrt_disc = copysign(s, delta_x) * sqrt_r(max(zero(T),
        abs2(inv_s * gc) - (inv_s * ga) * (inv_s * gt)))
    r = div_r(sqrt_disc + gt + gc, (ga - gt) - (sqrt_disc + sqrt_disc))
    tc = ((r > zero(T)) & !iszero(sqrt_disc)) ? (t + r * delta_x) :
         ((t > a) ? t_max : t_min)
    ts = secant_minimizer(a, ga, t, gt)
    t_next = bracketed ? ((abs(tc - t) < abs(ts - t)) ? tc : ts) :
             ((abs(tc - t) > abs(ts - t)) ? tc : ts)
    if bracketed
        mid = t + contraction_factor * (b - t)
        return (b > t) ? min(mid, t_next) : max(mid, t_next)
    else
        return clamp(t_next, t_min, t_max)
    end
end


function next_step_size(
    a::T, fa::T, ga::T,
    b::T, fb::T, gb::T,
    t::T, ft::T, gt::T,
    t_min::T, t_max::T, bracketed::Bool,
    contraction_factor::T,
) where {T}
    opposite_slope = (((gt > zero(T)) & (ga < zero(T))) |
                      ((gt < zero(T)) & (ga > zero(T))))
    t_next = if ft > fa
        bracketed = true
        higher_value_step_size(a, fa, ga, t, ft, gt)
    elseif opposite_slope
        bracketed = true
        opposite_slope_step_size(a, fa, ga, t, ft, gt)
    elseif abs(gt) < abs(ga)
        reduced_slope_step_size(a, fa, ga, b, t, ft, gt,
            t_min, t_max, bracketed, contraction_factor)
    elseif bracketed
        cubic_minimizer(b, fb, gb, t, ft, gt)
    else
        (t > a) ? t_max : t_min
    end
    if ft > fa
        b, fb, gb = t, ft, gt
    else
        if opposite_slope
            b, fb, gb = a, fa, ga
        end
        a, fa, ga = t, ft, gt
    end
    return (a, fa, ga, b, fb, gb, t_next, bracketed)
end


@inline function line_search_bounds(
    a::T, b::T, t::T, bracketed::Bool, initial::Bool,
    extrapolation_lower_bound::T,
    extrapolation_upper_bound::T,
) where {T}
    if initial
        return (zero(T), t + extrapolation_upper_bound * t)
    elseif bracketed
        return minmax(a, b)
    else
        return (
            t + extrapolation_lower_bound * (t - a),
            t + extrapolation_upper_bound * (t - a))
    end
end


@inline function line_search_update_bounds!(
    opt::ThomsonOptimizer{T},
    lane::Int,
    lsp::LineSearchParameters{T},
) where {T}
    _half = inv_r(one(T) + one(T))
    if opt.bracketed_host[lane]
        w = abs(opt.b_host[lane] - opt.a_host[lane])
        if w >= lsp.contraction_factor * opt.w0_host[lane]
            opt.step_sizes_host[lane] =
                _half * (opt.a_host[lane] + opt.b_host[lane])
        end
        opt.w0_host[lane] = opt.w_host[lane]
        opt.w_host[lane] = w
    end
    return opt
end


function line_search_initialize!(opt::ThomsonOptimizer{T}) where {T}
    fill!(opt.line_search_done_host, false)
    fill!(opt.bracketed_host, false)
    fill!(opt.modified_host, true)
    copyto!(opt.E0_host, opt.E_host)
    copyto!(opt.dE0_host, opt.dE_host)
    copyto!(opt.x0, opt.x)
    copyto!(opt.F0, opt.F)
    fill!(opt.a_host, zero(T))
    copyto!(opt.Ea_host, opt.E0_host)
    copyto!(opt.dEa_host, opt.dE0_host)
    fill!(opt.b_host, zero(T))
    copyto!(opt.Eb_host, opt.E0_host)
    copyto!(opt.dEb_host, opt.dE0_host)
    fill!(opt.w_host, zero(T))
    fill!(opt.w0_host, zero(T))
    return opt
end


@inline function line_search_update!(
    opt::ThomsonOptimizer{T},
    lane::Int,
    lsp::LineSearchParameters{T},
    t::T, t_min::T, t_max::T,
) where {T}
    opt.a_host[lane], opt.Ea_host[lane], opt.dEa_host[lane],
    opt.b_host[lane], opt.Eb_host[lane], opt.dEb_host[lane],
    opt.step_sizes_host[lane], opt.bracketed_host[lane] = next_step_size(
        opt.a_host[lane], opt.Ea_host[lane], opt.dEa_host[lane],
        opt.b_host[lane], opt.Eb_host[lane], opt.dEb_host[lane],
        t, opt.E_host[lane], opt.dE_host[lane],
        t_min, t_max, opt.bracketed_host[lane], lsp.contraction_factor)
    return opt
end


@inline function line_search_update_modified!(
    opt::ThomsonOptimizer{T},
    lane::Int,
    lsp::LineSearchParameters{T},
    t::T, t_min::T, t_max::T,
) where {T}
    c1_dE = lsp.c1 * opt.dE0_host[lane]
    opt.a_host[lane], Ea, dEa, opt.b_host[lane], Eb, dEb,
    opt.step_sizes_host[lane], opt.bracketed_host[lane] = next_step_size(
        opt.a_host[lane],
        opt.Ea_host[lane] - c1_dE * opt.a_host[lane],
        opt.dEa_host[lane] - c1_dE,
        opt.b_host[lane],
        opt.Eb_host[lane] - c1_dE * opt.b_host[lane],
        opt.dEb_host[lane] - c1_dE,
        t,
        opt.E_host[lane] - c1_dE * t,
        opt.dE_host[lane] - c1_dE,
        t_min, t_max, opt.bracketed_host[lane], lsp.contraction_factor)
    opt.Ea_host[lane] = Ea + c1_dE * opt.a_host[lane]
    opt.dEa_host[lane] = dEa + c1_dE
    opt.Eb_host[lane] = Eb + c1_dE * opt.b_host[lane]
    opt.dEb_host[lane] = dEb + c1_dE
    return opt
end


function line_search!(
    opt::ThomsonOptimizer{T},
    lsp::LineSearchParameters{T},
) where {T}
    line_search_initialize!(opt)
    for lane = 1:opt.num_lanes
        opt.step_sizes_host[lane] =
            opt.optimization_done_host[lane] ? zero(T) : one(T)
        t = opt.step_sizes_host[lane]
        t_min, t_max = line_search_bounds(
            zero(T), zero(T), t, false, true,
            lsp.extrapolation_lower_bound,
            lsp.extrapolation_upper_bound)
        opt.w_host[lane] = abs(t_max - t_min)
        opt.w0_host[lane] = opt.w_host[lane] + opt.w_host[lane]
    end

    while true
        evaluate_step!(opt)
        progress = false
        for lane = 1:opt.num_lanes

            if (opt.optimization_done_host[lane] |
                opt.line_search_done_host[lane])
                continue
            end

            decrease_target =
                opt.E0_host[lane] +
                lsp.c1 * opt.step_sizes_host[lane] * opt.dE0_host[lane]
            if ((opt.E_host[lane] <= decrease_target) &
                (abs(opt.dE_host[lane]) <= lsp.c2 * abs(opt.dE0_host[lane])))
                opt.line_search_done_host[lane] = true
                continue
            end

            if (opt.modified_host[lane] &
                (opt.E_host[lane] <= decrease_target) &
                (opt.dE_host[lane] >= zero(T)))
                opt.modified_host[lane] = false
            end

            t = opt.step_sizes_host[lane]
            initial = ((!opt.bracketed_host[lane]) &
                       iszero(opt.a_host[lane]) & iszero(opt.b_host[lane]))
            t_min, t_max = line_search_bounds(
                opt.a_host[lane], opt.b_host[lane], t,
                opt.bracketed_host[lane], initial,
                lsp.extrapolation_lower_bound,
                lsp.extrapolation_upper_bound)

            if (opt.modified_host[lane] &
                (decrease_target < opt.E_host[lane] <= opt.Ea_host[lane]))
                line_search_update_modified!(opt, lane, lsp, t, t_min, t_max)
            else
                line_search_update!(opt, lane, lsp, t, t_min, t_max)
            end

            line_search_update_bounds!(opt, lane, lsp)
            t_min, t_max = line_search_bounds(
                opt.a_host[lane], opt.b_host[lane], opt.step_sizes_host[lane],
                opt.bracketed_host[lane], false,
                lsp.extrapolation_lower_bound, lsp.extrapolation_upper_bound)

            if (opt.bracketed_host[lane] &
                !(t_min < opt.step_sizes_host[lane] < t_max))
                opt.step_sizes_host[lane] = opt.a_host[lane]
                opt.line_search_done_host[lane] = true
                continue
            end

            if !iszero(opt.step_sizes_host[lane])
                progress = true
            end

        end
        if !progress
            break
        end
    end

    return opt
end


################################################################################


export optimize!


function optimize!(
    opt::ThomsonOptimizer{T},
    lsp::LineSearchParameters{T},
) where {T}
    initialize!(opt)
    while true
        line_search!(opt, lsp)
        accept_step!(opt)
        for lane = 1:opt.num_lanes
            if !opt.optimization_done_host[lane]
                progress = (opt.line_search_done_host[lane] &
                            (opt.E_host[lane] < opt.E0_host[lane]))
                if !progress
                    opt.optimization_done_host[lane] = true
                end
            end
        end
        if all(opt.optimization_done_host)
            break
        end
        lbfgs!(opt)
        reset_history = false
        for lane = 1:opt.num_lanes
            if ((!opt.optimization_done_host[lane]) &
                (!signbit(opt.dE_host[lane])))
                reset_lbfgs_history!(opt, lane)
                reset_history = true
            end
        end
        if reset_history
            batched_dot_product!(opt.temp_lane, opt.temp_particle,
                opt.d, opt.F, opt.num_particles, opt.num_lanes,
                opt.block_sizes.dot, opt.block_sizes.dot_sum)
            copyto_negate!(opt.dE_host, opt.temp_lane)
        end
        opt.num_iterations[] += 1
    end
    return opt
end


################################################################################

end # module MinimalEnergy
