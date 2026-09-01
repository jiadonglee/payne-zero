# Literature check: grey, hydrostatic, and convective starting atmospheres

Date: 2026-08-31

## Question

Did earlier one-dimensional stellar-atmosphere calculations use hydrostatic
grey models with a convective correction, and why is the manuscript's exact
choice

\[
m_{\rm seed}=m_{\rm grey},\qquad T_{\rm seed}=T_{\rm conv}
\]

with no re-integration of column mass after changing the temperature not a
standard initialization?

## Main finding

The premise that earlier work did not use such initialization is incorrect.
Grey hydrostatic starting atmospheres, including convective adjustment, are
classical. What is unusual in the present work is narrower: the deliberate
decoupling that retains the fully constructed grey column-mass coordinate
after replacing the temperature profile, followed by recomputation of opacity
without re-integrating the mass coordinate.

No checked primary source explicitly advocates that exact decoupled choice.
This is an absence in the sources checked, not proof that it has never been
used.

## Evidence

### MARCS

Gustafsson et al. (2008) state that a calculation started from scratch
normally uses a grey starting model in the radiative region. If the first
model is convectively unstable, its temperature gradient is directly reduced
until the convective flux no longer exceeds the prescribed total flux.

The same paper also says that production grids usually start from a nearby
converged atmosphere, optionally scaling \(T(\tau_{\rm Ross})\) for a modest
change in effective temperature. This is faster and more reliable than a cold
grey start.

The paper reports a specific convergence problem in a band running roughly
from \(T_{\rm eff}=6750\) K, \(\log g=3\), to \(T_{\rm eff}=8000\) K,
\(\log g=5\). Convectively unstable regions alternate between thin zones and
zones extending below the model boundary. This is directly relevant to the
present concentration of physical-initializer failures around
\(7000\)--\(9000\) K.

### TLUSTY and coolTLUSTY

The TLUSTY guide says that column masses can either be read from an input
model or constructed in a starting LTE-grey atmosphere.

Hubeny (2017) gives the construction explicitly. It sets a Rosseland optical
depth grid, adopts a grey \(T(\tau)\), solves hydrostatic equilibrium while
recomputing Rosseland opacity, and obtains \(m=P/g\). If convection is
present, it computes the radiative and adiabatic gradients, solves for the
mixing-length gradient, and continues the integration with \(T\) as a
function of pressure. Its thermodynamic derivatives include ionization and
partition-function effects.

This is very close to the present physical ingredients, but it is a coupled
construction: temperature, pressure, opacity, and depth are integrated
together. It does not document the present choice of first completing the
grey mass coordinate and then freezing it while replacing only temperature.

### ATLAS

Hubeny's Appendix C identifies its initial-grey construction as closely
following Kurucz (1970). Current ATLAS12 instructions normally take an
existing ATLAS9 or ATLAS12 atmosphere as the input model and scale it to the
new stellar parameters before iteration. Thus ATLAS practice also favours a
nearby converged atmosphere when one is available.

### PHOENIX

The PHOENIX library of Husser et al. (2013) began with 766 pre-existing
atmospheres and extended the grid step by step, using an existing neighbouring
model as the starting point. This reflects the same practical strategy as
MARCS: use a close converged model instead of repeatedly starting from a grey
approximation.

## Why the exact decoupled choice was not standard

1. **Its components were already classical.** Grey radiative structure,
   hydrostatic integration, ionization-aware thermodynamics, and convective
   adjustment were already available. There was no missing general physical
   initializer to invent.

2. **Classical constructors preferred internal consistency.** On a Rosseland
   grid, \(d\tau_{\rm R}=\kappa_{\rm R}\,dm\). Changing \(T\) changes ionization
   and \(\kappa_{\rm R}\). Holding \(m\) fixed therefore makes the initial
   \(\tau_{\rm R}\), opacity, and mass relation temporarily inconsistent.
   Traditional algorithms instead update pressure or mass together with
   temperature and opacity, or solve the structural equations simultaneously.

3. **Nearby converged models are better starts.** Classical atmosphere work
   commonly generated grids in ordered sequences. A neighbouring model was
   already available and normally lay much closer to the final solution than
   a grey construction.

4. **The difficult physics is strongly coupled.** Molecular equilibrium,
   ionization, opacity, and the location of convective zones respond
   nonlinearly to temperature and pressure. Both MARCS and the present
   experiment show that convection-transition regions can be numerically
   difficult.

5. **The historical objective was different.** Traditional codes sought the
   fastest route to a grid of converged atmospheres. The present experiment
   asks a different question: whether an on-demand atmosphere can be
   initialized without an atmosphere emulator, and whether preserving one
   approximate coordinate is more robust than enforcing approximate
   self-consistency.

## Implication for the manuscript

The paper should not imply that a hydrostatic grey-convective start itself is
new. The defensible contribution is the controlled comparison of:

- a learned two-field initializer;
- a classical-style coupled physical construction; and
- the empirically selected decoupled variant that preserves
  \(m_{\rm grey}\).

The \(7000\)--\(9000\) K failure concentration should be connected to the
MARCS convergence band, with cautious language because the two solvers and
sample definitions differ.

## Primary sources

1. Gustafsson, B., Edvardsson, B., Eriksson, K., Jørgensen, U. G.,
   Nordlund, Å., & Plez, B. 2008, *A grid of MARCS model atmospheres for
   late-type stars. I. Methods and general properties*, A&A, 486, 951.
   DOI: https://doi.org/10.1051/0004-6361:200809724
   PDF: https://www.aanda.org/articles/aa/pdf/2008/30/aa09724-08.pdf

2. Hubeny, I. 2017, *Model atmospheres of sub-stellar mass objects*, MNRAS,
   469, 841. DOI: https://doi.org/10.1093/mnras/stx758
   Article: https://academic.oup.com/mnras/article/469/1/841/3092374

3. Hubeny, I. & Lanz, T., *TLUSTY User's Guide, version 202*.
   https://tlusty.oca.eu/tlusty/Tlusty2002/pdf/tlguide202.pdf

4. Husser, T.-O. et al. 2013, *A new extensive library of PHOENIX stellar
   atmospheres and synthetic spectra*, A&A, 553, A6.
   DOI: https://doi.org/10.1051/0004-6361/201219058
   PDF: https://www.aanda.org/articles/aa/pdf/2013/05/aa19058-12.pdf

5. Castelli, F., official ATLAS12 run instructions and source links.
   https://wwwuser.oats.inaf.it/fiorella.castelli/sources/atlas12.html

6. Kurucz, R. L. 1970, *ATLAS: A Computer Program for Calculating Model
   Stellar Atmospheres*, SAO Special Report 309.
   Bibliographic record:
   https://1535.sydneyplus.com/genieplus/final/ViewRecord.aspx?record=aa3e6e93-22b3-4882-9dda-4966f2ef2a2e&template=Books
