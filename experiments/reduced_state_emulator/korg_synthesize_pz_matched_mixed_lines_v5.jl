using DelimitedFiles
using HDF5
using Korg
using Printf

const KBOLTZ = Korg.kboltz_cgs
const REFERENCE_WAVELENGTH_CM = 5000.0e-8
const LINE_BUFFER_A = 1000.0
const MOLECULES = Dict(
    240 => "H2", 246 => "CH", 258 => "OH", 264 => "C2", 270 => "CN",
    300 => "MgH", 324 => "AlO", 342 => "CaH", 366 => "TiO",
    372 => "VO", 492 => "NaH",
)

function planar_from_payne(path, abundances)
    values = readdlm(path, '\t', Float64; skipstart=1)
    T, ne, Pgas, rho, m = eachcol(values)
    n = Pgas ./ (KBOLTZ .* T)
    z = zeros(Float64, length(T))
    for i in 2:length(T)
        z[i] = z[i - 1] - 0.5 * (m[i] - m[i - 1]) * (1 / rho[i] + 1 / rho[i - 1])
    end
    tau = zeros(Float64, length(T))
    tau[1] = 1e-8
    alpha_previous = 0.0
    for i in eachindex(T)
        ne_eq, number_densities = Korg.chemical_equilibrium(
            T[i], n[i], ne[i], abundances,
            Korg.ionization_energies, Korg.default_partition_funcs,
            Korg.default_log_equilibrium_constants;
            electron_number_density_warn_threshold=Inf,
        )
        alpha5000 = Korg.ContinuumAbsorption.total_continuum_absorption(
            [Korg.c_cgs / REFERENCE_WAVELENGTH_CM], T[i], ne_eq,
            number_densities, Korg.default_partition_funcs,
        )[1]
        if i > 1
            tau[i] = tau[i - 1] + 0.5 * (alpha5000 + alpha_previous) * (z[i - 1] - z[i])
        end
        alpha_previous = alpha5000
    end
    layers = [Korg.PlanarAtmosphereLayer(tau[i], z[i], T[i], ne[i], n[i]) for i in eachindex(T)]
    Korg.PlanarAtmosphere(layers, REFERENCE_WAVELENGTH_CM)
end

function load_lines(path)
    values = readdlm(path, '\t', Float64; skipstart=1)
    rows = map(eachrow(values)) do row
        wavelength_nm, loggf, kind, species_code, ion_stage, E_lower_eV, gamma_rad, gamma_stark, log_gamma_vdw = row
        kind_int = round(Int, kind)
        code_int = round(Int, species_code)
        species = if kind_int == 0
            Korg.Species(@sprintf("%d.%02d", code_int, round(Int, ion_stage) - 1))
        else
            Korg.Species(MOLECULES[code_int])
        end
        line = Korg.Line(
            wavelength_nm * 10.0, loggf, species, E_lower_eV,
            gamma_rad, gamma_stark, log_gamma_vdw,
        )
        (line=line, kind=kind_int, source_species_code=code_int)
    end
    sort!(rows; by=row -> row.line.wl)
end

line_path, atmosphere_path, abundance_path, output_path, start_A_text, end_A_text, step_A_text = ARGS
abundances = vec(readdlm(abundance_path, '\t', Float64))
atmosphere = planar_from_payne(atmosphere_path, abundances)
rows = load_lines(line_path)
lines = [row.line for row in rows]
start_A = parse(Float64, start_A_text)
end_A = parse(Float64, end_A_text)
step_A = parse(Float64, step_A_text)
wavelengths = Korg.Wavelengths((start_A, end_A, step_A))
used_rows = rows
used_lines = [row.line for row in used_rows]
raw = Korg.synthesize(
    atmosphere, lines, 12.0 .+ log10.(abundances ./ abundances[1]), wavelengths;
    vmic=1.0, line_buffer=LINE_BUFFER_A, hydrogen_lines=false, return_cntm=true,
    use_internal_reference_linelist=true, I_scheme="linear_flux_only", tau_scheme="anchored",
)

mkpath(dirname(output_path))
h5open(output_path, "w") do handle
    handle["wavelength_vacuum_A"] = raw.wavelengths
    handle["flux_total"] = raw.flux
    handle["flux_continuum"] = raw.cntm
    handle["line_wavelength_nm"] = [line.wl * 1e7 for line in used_lines]
    handle["line_loggf"] = [line.log_gf for line in used_lines]
    handle["line_kind"] = [row.kind for row in used_rows]
    handle["line_source_species_code"] = [row.source_species_code for row in used_rows]
    handle["line_ion_stage"] = [row.kind == 0 ? Int(row.line.species.charge) + 1 : 0 for row in used_rows]
    handle["line_E_lower_eV"] = [line.E_lower for line in used_lines]
    handle["line_gamma_rad_s"] = [line.gamma_rad for line in used_lines]
    handle["line_gamma_stark_s"] = [line.gamma_stark for line in used_lines]
    handle["line_gamma_vdw_s"] = [line.vdW[1] for line in used_lines]
    attrs(handle)["input_line_count"] = length(lines)
    attrs(handle)["used_line_count"] = length(used_lines)
    attrs(handle)["molecular_line_count"] = count(row -> row.kind == 1, used_rows)
    attrs(handle)["atomic_line_count"] = count(row -> row.kind == 0, used_rows)
    attrs(handle)["hydrogen_lines"] = false
    attrs(handle)["use_internal_reference_linelist"] = true
    attrs(handle)["wavelength_system"] = "vacuum"
    attrs(handle)["sampling_A"] = step_A
end
