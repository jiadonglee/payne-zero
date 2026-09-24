#!/usr/bin/env julia
"""Korg.jl 1.0.1 model arms for experiments/mgiant_eso_bestfit_v1.py.

    julia -t 8 --project=<Korg.jl-1.0.1> mgiant_eso_bestfit_v1_korg.jl \
        --root results/mgiant_eso_bestfit_v1 \
        --arm marcs_nodes|korg_pzatm|korg_pzatm_nozro|marcs_dense \
        [--part k --parts n]

Every arm writes the normalized intrinsic spectrum (flux / continuum) on
Korg's 0.01 A vacuum grid over the shared synthesis window, with the GALAH
DR3 line list, vmic = 2 km/s and scaled-solar abundances ([alpha/M] = 0).

    marcs_nodes  interpolate_marcs at the 78 Payne-Zero node labels
    korg_pzatm   the 78 converged Payne-Zero atmospheres (planar, tau_5000
                 from Korg's own continuum opacity)
    korg_pzatm_nozro  as korg_pzatm, without the ZrO lines, which have no
                 species in the Payne Zero molecular equilibrium
    marcs_dense  interpolate_marcs on the regular grid in dense_labels.tsv
"""

using HDF5
using Korg
using Printf

const LAMBDA_VAC_A = (6475.0, 6747.0)
const VMIC = 2.0
const REFERENCE_WAVELENGTH_CM = 5000.0e-8

function option(name::String, default::String)
    for i in eachindex(ARGS)
        ARGS[i] == name && i < length(ARGS) && return ARGS[i + 1]
    end
    default
end

root = option("--root", "")
arm = option("--arm", "")
part = parse(Int, option("--part", "1"))
parts = parse(Int, option("--parts", "1"))
isempty(root) && error("--root is required")
arm in ("marcs_nodes", "korg_pzatm", "korg_pzatm_nozro", "marcs_dense") || error("unknown --arm $arm")

function read_table(path)
    lines = readlines(path)
    header = split(lines[1], '\t')
    [Dict(header .=> split(line, '\t')) for line in lines[2:end] if !isempty(strip(line))]
end

function planar_from_payne(path::AbstractString, A_X)
    values = reduce(vcat, [reshape(parse.(Float64, split(strip(l), '\t')), 1, :) for l in readlines(path)[2:end]])
    T, ne, Pgas, rho, m = (values[:, k] for k in 1:5)
    n = Pgas ./ (Korg.kboltz_cgs .* T)
    all(diff(m) .> 0) || error("column mass not increasing in $path")
    z = zeros(length(T))
    for i in 2:length(T)
        z[i] = z[i - 1] - 0.5 * (m[i] - m[i - 1]) * (1 / rho[i] + 1 / rho[i - 1])
    end
    abs_abundances = 10.0 .^ (A_X .- 12.0)
    abs_abundances ./= sum(abs_abundances)
    tau = zeros(length(T))
    tau[1] = 1e-8
    alpha_prev = 0.0
    for i in eachindex(T)
        ne_eq, n_dict = Korg.chemical_equilibrium(
            T[i], n[i], ne[i], abs_abundances, Korg.ionization_energies,
            Korg.default_partition_funcs, Korg.default_log_equilibrium_constants;
            electron_number_density_warn_threshold=Inf)
        alpha = Korg.ContinuumAbsorption.total_continuum_absorption(
            [Korg.c_cgs / REFERENCE_WAVELENGTH_CM], T[i], ne_eq, n_dict, Korg.default_partition_funcs)[1]
        i > 1 && (tau[i] = tau[i - 1] + 0.5 * (alpha + alpha_prev) * (z[i - 1] - z[i]))
        alpha_prev = alpha
    end
    Korg.PlanarAtmosphere([Korg.PlanarAtmosphereLayer(tau[i], z[i], T[i], ne[i], n[i]) for i in eachindex(T)],
                          REFERENCE_WAVELENGTH_CM)
end

rows = read_table(joinpath(root, arm == "marcs_dense" ? "dense_labels.tsv" : "node_labels.tsv"))
selected = [i for i in eachindex(rows) if (i - 1) % parts == part - 1]
out_dir = joinpath(root, "korg", arm)
mkpath(out_dir)
out_path = joinpath(out_dir, @sprintf("part%02d_of%02d.h5", part, parts))

galah = Korg.get_GALAH_DR3_linelist()
linelist = filter(l -> LAMBDA_VAC_A[1] - 10 <= l.wl * 1e8 <= LAMBDA_VAC_A[2] + 10, galah)
arm == "korg_pzatm_nozro" && (linelist = filter(l -> string(l.species) != "OZr", linelist))
@printf("%s part %d/%d: %d models, %d lines (%d molecular)\n", arm, part, parts, length(selected),
        length(linelist), count(l -> Korg.ismolecule(l.species), linelist))

wavelengths = nothing
flux = Matrix{Float32}(undef, 0, 0)
status = String[]
seconds = Float64[]
for (k, i) in enumerate(selected)
    row = rows[i]
    Teff = parse(Float64, row["teff"]); logg = parse(Float64, row["logg"]); M_H = parse(Float64, row["m_h"])
    A_X = Korg.format_A_X(M_H)
    started = time()
    result = try
        atm = arm in ("korg_pzatm", "korg_pzatm_nozro") ? planar_from_payne(row["atmosphere_tsv"], A_X) :
              Korg.interpolate_marcs(Teff, logg, A_X)
        Korg.synthesize(atm, linelist, A_X, LAMBDA_VAC_A; vmic=VMIC)
    catch err
        @printf("  %s failed: %s\n", row["model_id"], sprint(showerror, err)[1:min(end, 200)])
        nothing
    end
    push!(seconds, time() - started)
    if result === nothing
        push!(status, "failed")
        continue
    end
    global wavelengths
    if wavelengths === nothing
        wavelengths = result.wavelengths
        global flux = fill(NaN32, length(wavelengths), length(selected))
    end
    flux[:, k] = Float32.(result.flux ./ result.cntm)
    push!(status, "ok")
    k % 20 == 0 && @printf("  %d/%d done (%.1f s last)\n", k, length(selected), seconds[end])
end

h5open(out_path * ".partial", "w") do h
    h["wavelength_vacuum_A"] = collect(wavelengths)
    h["normalized_flux"] = flux
    h["model_id"] = [rows[i]["model_id"] for i in selected]
    h["status"] = status
    h["seconds"] = seconds
    attrs(h)["arm"] = arm
    attrs(h)["korg_version"] = string(pkgversion(Korg))
    attrs(h)["linelist"] = arm == "korg_pzatm_nozro" ? "GALAH DR3 (Korg bundled) without ZrO" : "GALAH DR3 (Korg bundled)"
    attrs(h)["vmic_km_s"] = VMIC
end
mv(out_path * ".partial", out_path; force=true)
@printf("wrote %s (%d ok / %d)\n", out_path, count(==("ok"), status), length(status))
