from typing import List

import paper_data_linking.data.parsers as parsers
from pathlib import Path
import pytest

from paper_data_linking.utils import read_local_pdf

PDF_NEEDS_OCR = Path(__file__).parent / "1995SoPh__162__357B.pdf"
PDF_NO_NEED = Path(__file__).parent / "arxiv.pdf"
PDF_AGU = Path(__file__).parent / "agu.pdf"


def test_pdf_to_text_with_ocr():
    file_path = Path(__file__).parent / "1995SoPh__162__357B.pdf"
    with open(file_path, 'rb') as file:
        pdf_bytes = file.read()

    text = parsers.pdf_to_text_with_ocr(pdf_bytes)
    print(text)


@pytest.mark.parametrize("infile", [
    PDF_NEEDS_OCR,
    PDF_NO_NEED,
    PDF_AGU,
])
def test_get_text(infile):
    content = read_local_pdf(infile)
    text = parsers.get_text(content)
    print(text)
    assert isinstance(text, str)


@pytest.mark.parametrize("infile", [
    PDF_NEEDS_OCR,
    PDF_NO_NEED,
    PDF_AGU,
])
def test_get_data(infile):
    content = read_local_pdf(infile)
    data = parsers.get_data(content)
    print(data)
    assert isinstance(data, List)

texts = [
    """1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B
1995SoPh..162..357B""",
    """arXiv:astro-ph/0507662v2  24 Aug 2005
MS19378, JULY 1, 2005
Preprint typeset using LATEX style emulateapj v. 6/22/04
CONFINED AND EJECTIVE ERUPTIONS OF KINK-UNSTABLE FLUX ROPES
T. TÖRÖK 1 AND B. KLIEM 2
Received 2005 March 4; accepted 2005 July 18
ABSTRACT
The ideal helical kink instability of a force-free coronal magnetic ﬂux rope, anchored in the photosphere, is
studied as a model for solar eruptions. Using the ﬂux rope model of Titov & Démoulin (1999) as the initial
condition in MHD simulations, both the development of helical shape and the rise proﬁle of a conﬁned (or
failed) ﬁlament eruption (on 2002 May 27) are reproduced in very good agreement with the observations. By
modifying the model such that the magnetic ﬁeld decreases more rapidly with height above the ﬂux rope, a
full (or ejective) eruption of the rope is obtained in very good agreement with the developing helical shape and
the exponential-to-linear rise proﬁle of a fast coronal mass ejection (CME) (on 2001 May 15). This conﬁrms
that the helical kink instability of a twisted magnetic ﬂux rope can be the mechanism of the initiation and the
initial driver of solar eruptions. The agreement of the simulations with properties that are characteristic of many
eruptions suggests that they are often triggered by the kink instability. The decrease of the overlying ﬁeld with
height is a main factor in deciding whether the instability leads to a conﬁned event or to a CME.
Subject headings: Instabilities – MHD – Sun: corona – Sun: ﬂares – Sun: coronal mass ejections (CMEs)
1. INTRODUCTION
Large-scale solar eruptions occur as ﬂares, ﬁlament (or
prominence) eruptions, and coronal mass ejections (CMEs).
Despite their different observational appearance, it is believed
that these phenomena are manifestations of the same physical
processes, which involve the disruption of the coronal mag-
netic ﬁeld. Indeed, in the largest eruptions (eruptive ﬂares)
usually all three phenomena are observed. The theory of the
main phase of such events, referred to as the “standard model”
of eruptive ﬂares (e.g., Shibata 1999), is quite well estab-
lished. However, their initiation as well as the mechanism of
upward acceleration are still unclear. A variety of theoretical
models have been proposed to explain the impulsive onset and
initial evolution of solar eruptions (see, e.g., Forbes 2000).
Here we focus on a ﬂux rope instability model. This is mo-
tivated by the observation that solar eruptions often show the
phenomenology of a loop-shaped magnetic ﬂux system with
ﬁxed footpoints at the coronal base and signatures of magnetic
twist. Furthermore, erupting ﬁlaments very often develop a
clearly helical axis shape in the course of the eruption, which
is the characteristic property of the helical kink instability of a
twisted magnetic ﬂux rope. The instability occurs if the twist,
a measure of the winding of the ﬁeld lines about the ﬂux rope
axis, exceeds a critical value (Hood & Priest 1981).
In coronal applications, the simplifying assumption of
straight, cylindrically symmetric ﬂux ropes has nearly always
been used so far. Only very recently, Török, Kliem, & Titov
(2004, hereafter Paper I) performed the ﬁrst detailed study
of the kink instability of an arched ﬂux rope, line-tied to the
photosphere, using the analytical model of a force-free coro-
nal ﬂux rope developed by Titov & Démoulin (1999, hereafter
TD) as the initial condition in 3D ideal MHD simulations.
They have shown that this model relaxes to a numerical equi-
librium very close to the analytical expressions in the case of
subcritical twist and that the helical kink instability develops
for supercritical twist (see also Fan & Gibson 2003, 2004).
1 Mullard Space Science Laboratory, University College London, Holm-
bury St. Mary, Dorking, Surrey RH5 6NT, UK; tt@mssl.ucl.ac.uk
2 Astrophysical Institute Potsdam, An der Sternwarte 16, 14482 Potsdam,
Germany; bkliem@aip.de
In the course of the instability, a helical current sheet,
wrapped around the kinking and rising ﬂux rope where it
pushes into the surrounding ﬁeld, and a vertical current sheet
below the rope (which has no counterpart in the cylindrically
symmetric case) are formed. A vertical current sheet below
rising unstable magnetic ﬂux is the central element in the stan-
dard model of eruptive solar ﬂares. Further essential features
of solar eruptions could be reproduced in the simulations, as
for example the formation of transient soft X-ray sigmoids
(Kliem, Titov, & Török 2004, Paper II). However, a full erup-
tion of the conﬁguration has not yet been obtained; the ﬂux
rope reached an elevation of only about twice its initial height.
Here we present further developments of these simulations
to substantiate our suggestion in Papers I and II that the kink
instability of a coronal magnetic ﬂux rope is a possible trig-
ger mechanism of solar eruptions. The instability was ﬁrst
suggested as the trigger of (conﬁned and ejective) prominence
eruptions by Sakurai (1976), but has recently been generally
regarded as a possible explanation only for conﬁned events
(e.g., Gerrard & Hood 2003). The new simulations show that
the instability can also trigger full eruptions.
2. NUMERICAL MODEL
We integrate the compressible ideal MHD equations using
the simplifying assumptions of vanishing plasma-beta, β = 0,
and vanishing gravity, which are identical to Eqs. (2–5) in Pa-
per I. Setting β = 0 is usually a very good approximation in the
lower and middle corona of active regions, the source region
of most eruptions, where estimates yield β ∼ 10−3...10−2.
Both the pressure gradient force and the gravity force inﬂu-
ence the rise characteristics of the unstable ﬂux rope and the
energy partition in the development of the instability. How-
ever, the basic characteristics of the instability are well de-
scribed by the equations used whenever the Lorentz force
dominates, as is the case in the initial phase of solar erup-
tions. Magnetic reconnection occurs in the simulations due to
numerical diffusion if current sheets steepen sufﬁciently.
As in Papers I and II, we use the approximate analytical
force-free equilibrium of an arched, line-tied, and twisted ﬂux
rope by TD as the initial condition for the magnetic ﬁeld.
The ﬂux rope is modelled by the upper section of a toroidal
2
Török & Kliem
FIG. 1.— Left: TRACE 195 Å images of the conﬁned ﬁlament eruption
on 2002 May 27. Right: magnetic ﬁeld lines outlining the core of the kink-
unstable ﬂux rope (with start points in the bottom plane at circles of radius
b/3) at t = 0, 24, and 37. The central part of the box (a volume of size 43) is
shown, and the magnetogram, Bz(x,y,0,t), is included.
ring current, partly submerged below the photosphere, whose
Lorentz self-force is balanced by a pair of ﬁctitious subpho-
tospheric magnetic charges. A ﬁctitious subphotospheric line
current at the toroidal symmetry axis is included to achieve a
ﬁnite twist everywhere in the system. See TD for a detailed
description of the model.
The initial density distribution can be freely speciﬁed; we
choose it such that the Alfvén velocity in the volume sur-
rounding the ﬂux rope decreases slowly with height: ρ0 ∝
|B0(x)|3/2 (see Fig. 3 below). The system is at rest at t = 0,
except for a small upward velocity perturbation, which is lo-
calized at the ﬂux rope apex in a sphere of radius equal to the
minor radius, b, of the rope. Lengths, velocities, and times are
normalized, respectively, by the initial ﬂux rope apex height,
h0, the initial Alfvén velocity at the apex, vA0, and the corre-
sponding Alfvén time, τA = h0/vA0.
3. SIMULATION OF A CONFINED ERUPTION
The eruption of an active region ﬁlament (on 2002 May 27),
which was accompanied by an M2 ﬂare but did not lead to a
CME, was described by Ji et al. (2003). The ﬁlament started
to rise rapidly and developed a clear helical shape, as is often
observed; however, the ascent was terminated at a projected
height of ≈ 80 Mm (Figs. 1, 2). Such conﬁned ﬁlament erup-
tions are not uncommon (Rust 2003).
 
 
 
 
 
 
0
1
2
3
4
0
1
2
3
4
0
2
4
6
8
h(t)
h [104 km]
0
10
20
30
40
50
t  [τA]
0.0
0.1
0.0
0.1
0
1
2
3
u(t)
u [102 km s-1]
18:01
 
18:03
 
18:05
 
18:07
 
18:09
 UT
FIG. 2.— Comparison of height, h(t), and velocity, u(t), of the ﬂux rope
apex in the simulation (solid lines; initial perturbation is dotted) with the
corresponding values of the ﬁlament eruption in Fig. 1 (data from Fig. 3
of Ji et al. [2003]) overplotted as diamonds (the height data observed before
18:04 UT were smoothed here, resulting in reduced velocity scatter). See
text and Table 1 for the scaling of the dimensionless simulation variables
(left axes) to the observed values (right axes).
In order to model this event, we consider a kink-unstable
conﬁguration which is very similar to the case of an average
ﬂux rope twist of 4.9π studied in detail in Paper I. The line
current is reduced by about one third to enable a higher rise,
but the average twist, Φ = 5.0π here, is kept to reproduce the
helical shape. (In the TD model, this ﬁxes the minor radius
to b = 0.29, 8 percent larger than in the reference run in Pa-
per I.) The sign of the line current is chosen to be positive
to conform to the apparent positive (right-handed) helicity of
the observed ﬁlament. Furthermore, we increase the numeri-
cal diffusion and prevent the density from becoming negative,
which permits us to follow the evolution of the system for a
considerably longer time. Otherwise, the magnetic conﬁgura-
tion and the numerical settings are the same as in Paper I.
As in our previous simulations, the upwardly directed kink
instability leads to the ascent and helical deformation of the
ﬂux rope as well as to the formation of current sheets (see
Fig. 3 in Paper I). In Fig. 1 we compare the evolution of the he-
lical shape of the ﬂux rope with Transition Region and Coro-
nal Explorer (TRACE) observations of the ﬁlament eruption.
The evolution is remarkably similar. Figure 2 shows that the
principal features of the observed rise are also matched. After
an exponential rise the ﬂux rope comes to a stop at ≈ 3.5 h0.
A ﬁrst deceleration occurs as the current density in the heli-
cal current sheet above the apex begins to exceed the current
density in the ﬂux rope (t > 22). The subsequent upward push
(t > 30) results from the reconnection outﬂow in the vertical
current sheet below the rope. Finally, the rise is terminated by
the onset of magnetic reconnection in the current sheet above
the rope, which progressively cuts the rope ﬁeld lines (t ≳ 33).
The reconnection outﬂows expand the top part of the rope in
lateral direction, as seen both in observation and simulation.
Using the scaling to dimensional values given in Table 1,
good quantitative agreement with the rise proﬁle is obtained
(Fig. 2), and the release of magnetic energy in this run of
5 percent corresponds to 1031 erg, a reasonable value for a
conﬁned M2-class ﬂare.
The agreement between the observations of the event and
our simulation conﬁrms the long-held conjectures that the de-
velopment of strongly helical axis shapes in the course of
eruptions can be regarded as an indication of the kink instabil-
Eruptions of kink-unstable ﬂux ropes
3
TABLE 1
PARAMETER SETTINGS
Simulation Parameters
Scaling Parameters
Sect.
Φ/π
b
η(2)
L
h0
τA
|B0(h0)|
W
(Mm)
(s)
(G)
(erg)
3
5.0
0.29
0.83
10
23
11.5
200
1031
4
−5.0
0.33
1.54
32
115
111
10-40
1031–32
NOTE. — The expression η(z) = −zd lnBex(0,0,z,0)/dz is the ‘decay
index’ of the ‘external’ ﬁeld (excluding the contribution by the ring current),
L is the box size, and W is the released magnetic energy. The runs are equal
in grid resolution in the central part of the box, ∆ = 0.02, major rope radius,
R = 1.83, and distance of the ﬁctitious magnetic charges from the z axis,
l = 0.83.
ity of a twisted ﬂux rope and that the frequently observed heli-
cal ﬁne structures in erupting ﬁlaments and prominences out-
line twisted ﬁelds. Furthermore, it shows that ﬂux ropes with
substantial twist can exist or be formed in the solar corona at
the onset of, or prior to, eruptions.
4. SIMULATION OF AN EJECTIVE ERUPTION (CME)
The full eruption of the kink-unstable ﬂux rope in the TD
model is prevented by the strong overlying ﬁeld, which is
dominated by the line current. It is possible to obtain an erup-
tive behaviour of the ﬂux rope by removing the line current;
however, such a modiﬁcation leads to an inﬁnite number of
ﬁeld line turns at the surface of the ﬂux rope (Roussev et al.
2003).
In order to avoid this problem, we replaced the
line current by a pair of subphotospheric dipoles (as used in
Török & Kliem 2003). The position of the dipoles is chosen
such that the ﬁeld lines of the dipole pair passing through the
ﬂux rope match the curvature of the rope as closely as possi-
ble. The resulting equilibrium yields a ﬁnite twist everywhere
in the system, but the magnetic ﬁeld overlying the ﬂux rope
now decreases signiﬁcantly faster with height than in the orig-
inal TD model (Fig. 3). By varying the dipole moments or the
minor radius b, one can adjust the average twist within the
ﬂux rope.
Choosing suitable dipole moments and b = 0.6, but oth-
erwise the same parameters of the TD model as in Sect. 3,
we ﬁrst checked that the modiﬁed conﬁguration relaxes to a
nearby stable equilibrium for subcritical twist, Φ = 2.7π. Next
a conﬁguration with supercricital twist, Φ = −5.0π, is consid-
ered, obtained by changing the minor radius to b = 0.33 and
reversing the dipole moments. The sign of the helicity corre-
sponds to the 2001 May 15 event discussed below; it has no
inﬂuence on the rise, h(t), of the ﬂux rope apex. The numer-
ical parameters of the simulation are the same as in Sect. 3,
except for a considerably larger simulation box and a smaller
level of numerical diffusion, which this system permitted.
The helical kink instability also develops in the modiﬁed
model. However, the ﬂux rope now exhibits a much stronger
expansion (Fig. 4), which is not slowed down. The initially
exponential rise is followed by a rise with approximately con-
stant and locally super-Alfvénic velocity, until the rope en-
counters the top of the simulation box (at t ≈ 80). The helical
current sheet remains very weak on top of the ﬂux rope apex
and no signiﬁcant amount of reconnection occurs here. On the
other hand, the vertical current sheet now steepens in a large
height range. Magnetic reconnection commences in this sheet
at the beginning of the exponential phase and rises in tandem
with the ascent of the ﬂux rope, particularly closely during the
0
5
10
15
z
0.001
0.010
0.100
1.000
B0, vA
FIG. 3.— Normalized initial magnetic ﬁeld strength (thick lines) and Alfvén
velocity (thin lines) vs. height for the conﬁgurations described in Sect. 3
(original TD model; dashed lines) and Sect. 4 (modiﬁed TD model; solid
lines).
exponential phase. Since the ﬂux rope expands continuously
during this phase (instead of being compressed by the upward
reconnection outﬂow below it) and moves away from the fore-
front of the outﬂow region afterwards, the ideal instability of
the ﬂux rope appears to be the driver of the closely coupled
processes. Cusp-shaped ﬁeld lines are formed throughout the
evolution (Fig. 4) but most prominently in the late phase, in
agreement with soft X-ray observations of eruptive ﬂares.
The full eruption of the ﬂux rope in the modiﬁed TD model
must be enabled by the weaker overlying ﬁeld, since all other
parameters are identical, or very close, to Sect. 3.
In Fig. 5 we compare the rise of the ﬂux rope apex in the
simulation with the rise of the apex of a well observed erup-
tive prominence on 2001 May 15, which occurred in a spot-
less region slightly behind the limb and led to a fast CME
(peak velocity of leading edge ≈ 1200 km s−1) and a long-
duration ﬂare (X-ray class C4). As described by Mariˇci´c et al.
(2004), the eruptive prominence developed a helical shape,
analogous to the middle panels in Fig. 1 with reversed hand-
edness, and it showed the typical rise characteristics of a fast
CME (initially exponential or exponential-like rise, followed
by approximately linear rise; Vršnak 2001). For this com-
parison, we ﬁrst scaled the Alfvén time such that the dura-
tion of the exponential rise phase is matched, τA = 111 s,
and shifted the time axis accordingly. Then we scaled the
length unit such that the apex height at the point of peak
acceleration in the simulation (t = 21) equals the height of
the prominence at the resulting observation time (see bottom
axis), i.e., h0 = 115 Mm. This ﬁxes the scaling of the velocity
and acceleration amplitudes. Apart from a somewhat more
gradual decrease of the observed acceleration after the peak,
excellent qualitative and quantitative agreement is obtained,
demonstrating (as in Sect. 3) that the kink instability yields
the growth rate required by the observed rise proﬁle for the
twist indicated by the observed helical shape. The slight dif-
ference in the late acceleration proﬁle may have many origins,
for example, a different height proﬁle of the ﬁeld strength, or
a simultaneous expansion of the overlying ﬁeld enforced by
photospheric ﬂows in the observed event.
The simulation shows a strong magnetic energy release,
25 percent of the initial value, which agrees with the mag-
nitude observed in ejective solar eruptions (Forbes 2000;
Emslie et al. 2004). The considered event may have released
magnetic energy of order ∼ (1031–1032) erg (the low X-ray
class resulted from footpoint occultation). Using the above
scaling for h0, this energy release is reproduced for B0 ∼
4
Török & Kliem
FIG. 4.— Magnetic ﬁeld lines of the kink-unstable modiﬁed TD model at t = 0 (left), t = 30 (center), and t = 43 (right). The magnetogram, Bz(x,y,0,t), is
included. Field lines started at a circle of radius b/3 in the bottom plane show the core of the ﬂux rope. Additional green ﬁeld lines, also with identical start
points in all panels, indicate the formation of “post ﬂare loops” with a cusp by reconnection. The hyperbolic point of the ﬁeld at the z-axis (magnetic X-point)
lies at z ≈ 0.2, 0.6, and 1.1, respectively.
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
1.00
10.0
h(t)
0.1  
1.0  
h [RSun]
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
0.01
0.10
1.00
u(t)
101 
102 
103 
u [km s-1]
0
10
20
30
40
50
 
 
 
 
 
0
10
20
30
40
50
t  [τA]
 
 
 
 
 
 
0.00
 
0.02
 
0.04
a(t)
0.0 
 
0.2 
 
0.4 
a [km s-2]
18.0
18.5
19.0
UT
FIG. 5.— Comparison of the simulation in Sect. 4 with the CME on 2001
May 15 in the same format as in Fig. 2, including the acceleration a(t). Dia-
monds show the apex motion of the erupting prominence/the CME core (data
are from Fig. 6a, c of Mariˇci´c et al. [2004]).
(10–40) G, consistent with expected averages of the coronal
ﬁeld strength over the large length scales involved.
The simulation could also be scaled to a ﬁlament eruption
that was associated with an X-class ﬂare and a very fast CME
(on 2004 November 10; see Williams et al. 2005).
A line-tied ﬂux rope was found to erupt in a few previous
simulations (Amari et al. 2000, 2003a,b). The present simu-
lations, through their agreement with characteristic properties
of solar eruptions, identify a mechanism for the process, con-
ﬁrming the original suggestion by Sakurai (1976). They also
demonstrate the importance of the height dependence of the
overlying ﬁeld for the evolution of the instability into a CME,
while Amari et al. (2003b) found the amount of magnetic he-
licity to be essential. Since Amari et al. built up the helicity
by rotating the main photospheric polarities, which simulta-
neously expands the overlying ﬁeld (Török & Kliem 2003),
both results are fully consistent with each other.
5. CONCLUSIONS
Our MHD simulations of the kink instability of a coronal
magnetic ﬂux rope reproduce essential properties—an ini-
tially exponential rise with the rapid development of a heli-
cal shape—of two well observed solar eruptions, one of them
conﬁned, the other ejective. The subsequent approximately
linear rise of the ejective eruption is reproduced as well. Since
these features are characteristic properties of many solar erup-
tions (Vršnak 2001), we regard the kink instability of coronal
magnetic ﬂux ropes as the initiation mechanism and initial
driver of many such events. A sufﬁciently steep decrease of
the magnetic ﬁeld with height above the ﬂux rope permits the
process to evolve into a CME.
We acknowledge constructive comments by the referee and
thank H. Ji and B. Vršnak for the observation data in Figs. 2
and 5, respectively.
This work was supported by grants
50 OC 9706 (DLR), MA 1376/16-2 (DFG), and HPRN-CT-
2000-00153 (EU). The John von Neumann Institute for Com-
puting, Jülich granted computer time.
REFERENCES
Amari, T., et al. 2003a, ApJ, 585, 1073
Amari, T., et al. 2003b, ApJ, 595, 1231
Amari, T., Luciani, J. F., Mikic, Z., & Linker, J. 2000, ApJ, 529, L49
Emslie, A. G., et al. 2004, J. Geophys. Res., 109, A10104
Fan, Y., & Gibson, S. E. 2003, ApJ, 589, L105
Fan, Y., & Gibson, S. E. 2004, ApJ, 609, 1123
Forbes, T. G. 2000, J. Geophys. Res., 105, 23 153
Gerrard, C. L., & Hood, A. W. 2003, Sol. Phys., 214, 151
Hood, A. W., & Priest, E. R. 1981, Geophys. Astrophys. Fluid Dyn., 17, 297
Ji, H., et al. 2003, ApJ, 595, L135
Kliem, B., Titov, V. S., & Török, T. 2004, A&A, 413, L23 (Paper II)
Mariˇci´c, D., Vršnak, B., Stanger, A. L., & Veronig, A. 2004, Sol. Phys., 225,
337
Roussev, I. I., et al. 2003, ApJ, 588, L45
Rust, D. M. 2003, Adv. Space Res., 32, 1895
Sakurai, T. 1976, PASJ, 28, 177
Shibata, K. 1999, Ap&SS, 264, 129
Titov, V. S., & Démoulin, P. 1999, A&A, 351, 707 (TD)
Török, T., & Kliem, B. 2003, A&A, 406, 1043
Török, T., Kliem, B., & Titov V. S. 2004, A&A, 413, L27 (Paper I)
Eruptions of kink-unstable ﬂux ropes
5
Vršnak, B. 2001, J. Geophys. Res., 106, 25 249
Williams, D. R., Török, T., Démoulin, P., van Driel-Gesztelyi, L., & Kliem,
B. 2005, ApJ, 628, L163"""
]
texts_junk = [(texts[0], True), (texts[1], False)]


@pytest.mark.parametrize("text, is_junk", texts_junk)
def test_is_junk_pdf(text, is_junk):
    junk = parsers.is_junk_pdf(text)
    assert junk == is_junk
