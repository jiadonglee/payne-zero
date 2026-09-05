using DelimitedFiles
using HDF5
using Korg
using Printf

const KBOLTZ = Korg.kboltz_cgs
const REFERENCE_WAVELENGTH_CM = 5000.0e-8
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
    tau = zeros(Float64, length(T)); tau[1] = 1e-8
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

function load_compiled_kurucz(path)
    values = readdlm(path, '\t', Float64; skipstart=1)
    map(eachrow(values)) do row
        wavelength_nm, loggf, kind, species_code, ion_stage, E_lower_eV, gamma_rad, gamma_stark, log_gamma_vdw = row
        species = if round(Int, kind) == 0
            Korg.Species(@sprintf("%d.%02d", round(Int, species_code), round(Int, ion_stage) - 1))
        else
            Korg.Species(MOLECULES[round(Int, species_code)])
        end
        Korg.Line(wavelength_nm * 10.0, loggf, species, E_lower_eV,
                  gamma_rad, gamma_stark, log_gamma_vdw)
    end
end

function load_compiled_kurucz_atomic(path)
    values = readdlm(path, '\t', Float64; skipstart=1)
    map(eachrow(values)) do row
        wavelength_nm, loggf, atomic_number, ion_stage, E_lower_eV, gamma_rad, gamma_stark, log_gamma_vdw = row
        species = Korg.Species(@sprintf("%d.%02d", round(Int, atomic_number), round(Int, ion_stage) - 1))
        Korg.Line(wavelength_nm * 10.0, loggf, species, E_lower_eV,
                  gamma_rad, gamma_stark, log_gamma_vdw)
    end
end

mode, line_path, atmosphere_path, abundance_path, output_path,
start_A_text, end_A_text, step_A_text, vmic_text = ARGS
start_A = parse(Float64, start_A_text)
end_A = parse(Float64, end_A_text)
step_A = parse(Float64, step_A_text)
vmic = parse(Float64, vmic_text)
abundances = vec(readdlm(abundance_path, '\t', Float64))
atmosphere = planar_from_payne(atmosphere_path, abundances)

all_lines = if mode == "native"
    Korg.get_GALAH_DR3_linelist()
elseif mode == "native_atomic"
    Korg.get_VALD_solar_linelist()
elseif mode == "kurucz_atomic"
    load_compiled_kurucz_atomic(line_path)
else
    load_compiled_kurucz(line_path)
end
in_window(line) = (start_A - 10.0) <= line.wl * 1e8 <= (end_A + 10.0)
lines = filter(in_window, all_lines)
wavelengths = Korg.Wavelengths((start_A, end_A, step_A))
started = time()
raw = Korg.synthesize(
    atmosphere, lines, 12.0 .+ log10.(abundances ./ abundances[1]), wavelengths;
    vmic=vmic, line_buffer=10.0, hydrogen_lines=false, return_cntm=true,
    use_internal_reference_linelist=true, I_scheme="linear_flux_only", tau_scheme="anchored",
)

mkpath(dirname(output_path))
h5open(output_path, "w") do handle
    handle["wavelength_vacuum_A"] = raw.wavelengths
    handle["flux_total"] = raw.flux
    handle["flux_continuum"] = raw.cntm
    attrs(handle)["mode"] = mode
    attrs(handle)["source_line_count"] = length(all_lines)
    attrs(handle)["window_line_count"] = length(lines)
    attrs(handle)["atomic_line_count"] = count(line -> !Korg.ismolecule(line.species), lines)
    attrs(handle)["molecular_line_count"] = count(line -> Korg.ismolecule(line.species), lines)
    attrs(handle)["tio_line_count"] = count(line -> string(line.species) == "OTi", lines)
    attrs(handle)["seconds"] = time() - started
    attrs(handle)["wavelength_system"] = "vacuum"
end
