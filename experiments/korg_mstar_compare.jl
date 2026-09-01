#!/usr/bin/env julia
"""Korg.jl 1.0.1 comparison backend for the Payne-Zero M-star case.

The Python driver writes one fixed, converged Payne-Zero atmosphere per case.
This script constructs the Korg planar atmosphere from those columns, deriving
height from dm/rho and tau_ref from Korg's own 5000 A continuum opacity.  It
also evaluates Korg's independent MARCS interpolation with spherical=false.
"""

using DelimitedFiles
using HDF5
using Korg
using Printf
using Statistics

const KBOLTZ = Korg.kboltz_cgs
const REFERENCE_WAVELENGTH_A = 5000.0
const REFERENCE_WAVELENGTH_CM = REFERENCE_WAVELENGTH_A * 1e-8
const RESOLUTION = 20_000.0

function option(name::String, default::String)
    for i in eachindex(ARGS)
        if ARGS[i] == name && i < length(ARGS)
            return ARGS[i + 1]
        end
    end
    default
end

manifest_path = option("--manifest", "")
out_root = option("--out-dir", "")
mode = option("--mode", "smoke")
isempty(manifest_path) && error("--manifest is required")
isempty(out_root) && error("--out-dir is required")
mode in ("smoke", "full") || error("--mode must be smoke or full")

function read_manifest(path::String)
    rows = NamedTuple[]
    lines = readlines(path)
    length(lines) >= 2 || return rows
    fields = Symbol.(split(lines[1], '\t'))
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, '\t')
        length(values) == length(fields) || error("invalid manifest row: $line")
        row = Dict{Symbol,String}(fields .=> values)
        push!(rows, (
            case_id = row[:case_id],
            class = row[:class],
            Teff = parse(Float64, row[:effective_temperature]),
            logg = parse(Float64, row[:logg]),
            M_H = parse(Float64, row[:metallicity]),
            alpha_m = parse(Float64, row[:alpha_enhancement]),
            C_m = parse(Float64, row[:carbon_enhancement]),
            vmic = parse(Float64, row[:microturbulence_km_s]),
            product_path = row[:product_path],
            atmosphere_path = row[:atmosphere_path],
        ))
    end
    rows
end

function read_payne_columns(path::String)
    lines = readlines(path)
    length(lines) >= 3 || error("Payne-Zero atmosphere input is empty: $path")
    matrix = reduce(vcat, [reshape(parse.(Float64, split(strip(line), '\t')), 1, :) for line in lines[2:end]])
    size(matrix, 2) == 5 || error("expected five Payne-Zero columns in $path")
    matrix
end

function planar_from_payne(path::String)
    # columns: T, ne, Pgas, rho, m; dm/rho is a geometric layer thickness.
    values = read_payne_columns(path)
    T = values[:, 1]
    ne = values[:, 2]
    Pgas = values[:, 3]
    rho = values[:, 4]
    m = values[:, 5]
    n = Pgas ./ (KBOLTZ .* T)
    all(isfinite, T) && all(T .> 0) || error("invalid temperatures in $path")
    all(isfinite, rho) && all(rho .> 0) || error("invalid mass densities in $path")
    all(isfinite, m) && all(diff(m) .> 0) || error("column mass is not increasing in $path")

    z = zeros(Float64, length(T))
    for i in 2:length(T)
        dz = 0.5 * (m[i] - m[i - 1]) * (1 / rho[i] + 1 / rho[i - 1])
        z[i] = z[i - 1] - dz
    end

    A_X = Korg.format_A_X(0.0)
    abs_abundances = 10.0 .^ (A_X .- 12.0)
    abs_abundances ./= sum(abs_abundances)
    tau = zeros(Float64, length(T))
    tau[1] = 1e-8
    tau_alpha_prev = 0.0
    for i in 1:length(T)
        ne_eq, n_dict = Korg.chemical_equilibrium(
            T[i], n[i], ne[i], abs_abundances,
            Korg.ionization_energies, Korg.default_partition_funcs,
            Korg.default_log_equilibrium_constants;
            electron_number_density_warn_threshold=Inf,
        )
        alpha5000 = Korg.ContinuumAbsorption.total_continuum_absorption(
            [Korg.c_cgs / REFERENCE_WAVELENGTH_CM], T[i], ne_eq, n_dict,
            Korg.default_partition_funcs,
        )[1]
        isfinite(alpha5000) && alpha5000 > 0 || error("invalid Korg 5000 A continuum opacity at layer $i")
        if i > 1
            # z is ordered top-to-bottom and decreases, so -dz is positive.
            tau[i] = tau[i - 1] + 0.5 * (alpha5000 + tau_alpha_prev) * (z[i - 1] - z[i])
        end
        tau_alpha_prev = alpha5000
    end
    layers = [Korg.PlanarAtmosphereLayer(tau[i], z[i], T[i], ne[i], n[i]) for i in eachindex(T)]
    Korg.PlanarAtmosphere(layers, REFERENCE_WAVELENGTH_CM)
end

function wavelengths_for_mode(mode::String)
    mode == "smoke" ? (5000.0, 5020.0, 0.01) : (4000.0, 9000.0, 0.05)
end

function synthesize_and_save(atm, linelist, A_X, vmic::Float64, path::String, mode::String)
    mkpath(dirname(path))
    bounds = wavelengths_for_mode(mode)
    started = time()
    raw = Korg.synthesize(
        atm, linelist, A_X, bounds;
        vmic=vmic,
        hydrogen_lines=true,
        return_cntm=true,
        use_internal_reference_linelist=true,
        I_scheme="linear_flux_only",
        tau_scheme="anchored",
    )
    # Korg.synthesize is evaluated on a fine linear grid; apply the requested
    # constant resolving power before writing the comparison product.
    total = Korg.apply_LSF(raw.flux, raw.wavelengths, RESOLUTION)
    continuum = Korg.apply_LSF(raw.cntm, raw.wavelengths, RESOLUTION)
    normalized = total ./ max.(abs.(continuum), 1e-300)
    h5open(path, "w") do handle
        handle["wavelength_A"] = raw.wavelengths
        handle["flux_total"] = total
        handle["flux_continuum"] = continuum
        handle["normalized_flux"] = normalized
        attrs(handle)["mode"] = mode
        attrs(handle)["geometry"] = "planar"
        attrs(handle)["resolution"] = RESOLUTION
        attrs(handle)["seconds"] = time() - started
        attrs(handle)["linelist_lines"] = length(linelist)
        attrs(handle)["molecular_lines"] = count(line -> Korg.ismolecule(line.species), linelist)
    end
    @printf("wrote %s (%d lines, %d pixels)\n", path, length(linelist), length(raw.wavelengths))
end

rows = read_manifest(manifest_path)
isempty(rows) && error("no usable rows in $manifest_path")
all_lines = Korg.get_VALD_solar_linelist()
atomic_lines = filter(line -> !Korg.ismolecule(line.species), all_lines)
@printf("Korg 1.0.1: %d total VALD lines, %d atomic-only lines, %d molecular lines\n",
        length(all_lines), length(atomic_lines), length(all_lines) - length(atomic_lines))
A_X = Korg.format_A_X(0.0)

for row in rows
    same_atmosphere = planar_from_payne(row.atmosphere_path)
    independent_marcs = Korg.interpolate_marcs(
        row.Teff, row.logg, row.M_H, row.alpha_m, row.C_m;
        spherical=false,
        resampled_cubic_for_cool_dwarfs=true,
    )
    for (tag, linelist) in (("molecular", all_lines), ("atomic_only", atomic_lines))
        synthesize_and_save(
            same_atmosphere, linelist, A_X, row.vmic,
            joinpath(out_root, "same_atmosphere", tag, row.case_id * ".h5"), mode,
        )
        synthesize_and_save(
            independent_marcs, linelist, A_X, row.vmic,
            joinpath(out_root, "independent_marcs", tag, row.case_id * ".h5"), mode,
        )
    end
end
