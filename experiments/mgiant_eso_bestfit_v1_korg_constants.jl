# Korg 1.0.1 molecular data used to give Payne Zero a ZrO equilibrium entry:
# log10 K_p (partial-pressure form, cgs) and partition functions on a T grid.
using Korg, Printf
out = ARGS[1]
Ts = collect(1000.0:50.0:10000.0)
open(out, "w") do io
    println(io, "T\tlogKp_TiO\tlogKp_ZrO\tU_TiO\tU_ZrO\tU_Ti\tU_Zr\tU_O")
    for T in Ts
        lT = log(T)
        kp(s) = Korg.default_log_equilibrium_constants[Korg.Species(s)](lT)
        U(s) = Korg.default_partition_funcs[Korg.Species(s)](lT)
        @printf(io, "%.1f\t%.8f\t%.8f\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n", T, kp("TiO"), kp("ZrO"),
                U("TiO"), U("ZrO"), U("Ti I"), U("Zr I"), U("O I"))
    end
end
