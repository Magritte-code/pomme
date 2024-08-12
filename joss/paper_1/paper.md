---
title: 'pomme: Bayesian model reconstruction based on spectral line observations'

tags:
  - Python
  - astronomy
  - radiative transfer

authors:
  - name: Frederik De Ceuster^[Corresponding author.]
    orcid: 0000-0001-5887-8498
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Thomas Ceulemans
    orcid: 0000-0002-7808-9039
    affiliation: "1"
  - name: Leen Decin
    orcid: 0000-0002-5342-8612
    affiliation: "1"
  - name: Ta\"issa Danilovich
    orcid: 0000-0002-1283-6038
    affiliation: "2, 3, 1"
  - name: Jeremy Yates
    orcid: 0000-0003-1954-8749
    affiliation: "4"

affiliations:
 - name: Institute of Astronomy, Department of Physics & Astronomy, KU Leuven, Celestijnenlaan 200D, 3001 Leuven, Belgium
   index: 1
 - name: School of Physics \& Astronomy, Monash University, Clayton, Victoria, Australia
   index: 2
 - name: ARC Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D), Clayton, Victoria, Australia
   index: 3
 - name: Department of Computer Science, University College London, WC1E 6EA, London, United Kingdom
   index: 4
date: 1 March 2024
bibliography: paper.bib


# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal Supplement Series
---


# Summary
A typical problem in astronomy is that, for most of our observations, we are restricted to the plane of the sky.
As a result, these observations are always mere projections containing only partial information about the observed object.
Luckily, some frequency bands of the electromagnetic spectrum are optically thin, such that we receive radiation from the entire medium along the line of sight.
This means that, at least in principle, from the observed radiation, we can extract information about the physical and chemical conditions along the entire line of sight.
This is especially the case for spectral line radiation caused by transitions between the quantized energy levels of atoms and molecules in the medium.
Rotational transition lines are particularly interesting, since they are excited in many astrophysical environments and can easily be resolved individually.
Using spectral line observations, we can infer information about physical and chemical parameters, such as abundance of certain molecules, velocity, and temperature distributions.
To facilitate this, we built the [pomme]{.sc} Python package that helps users reconstruct 1D spherically symmetric or generic 3D models, based on astronomical spectral line observations.
A detailed description and validation of the methods can be found in [@DeCeuster2024].


# Statement of need
Spectral line observations are indispensible in astronomy.
Many line radiation transport solvers exist to solve the (forward) problem of determining what spectral line observations of a certain model would look like [see e.g. @DeCeuster2022].
However, far fewer tools exist to efficiently sovle the inverse problem of determining what model could correspond to certain spectral line observations.




<!-- [@DeCeuster:2022] -->

<!-- 
Recent high-resolution observations exposed the intricate and intrinsically 3D
morphologies of stellar winds around evolved stars [@Decin:2020]. The sheer amount of complexity that is
observed, makes it difficult to interpret the observations and necessitates the use of
3D hydrodynamics, chemistry and radiative transfer models to study their origin and
evolution [@ElMellah:2020; @Maes:2021; @Malfait:2021]. Their intricate
morpho-kinematics, moreover, makes their appearance in (synthetic) observations far from evident
(see e.g.\ the intricate structures in \autoref{fig:example}). Therefore, to study these and other complex
morpho-kinematical objects, it is essential to understand how their models would
appear in observations. This can be achieved, by creating synthetic observations
with Magritte.
Examples and analytic as well as cross-code benchmarks can be found in the documentation and in [@DeCeuster:2019; @DeCeuster:2020].

![Example of a synthetic observation of the CO($v=0$, $J=1-0$) transition, created with Magritte for a hydrodynamics model of an asymptotic giant branch (AGB) star, as it is perturbed by a companion [this is model \textsc{v10e50} in @Malfait:2021]. \label{fig:example}](example.png) -->


<!-- # Method
The Bayesian reconstruction algorithm aims to maximise the posterior distribution, $p(\boldsymbol{m}|\boldsymbol{m})$, of a model $\boldsymbol{m}$, given an observation $\boldsymbol{o}$.
Using Bayes' rule
$$
$$
We provide several prior -->



# Acknowledgements

FDC is a Postdoctoral Research Fellow of the Research Foundation - Flanders (FWO), grant number 1253223N, and was previously supported for this research by a KU Leuven Postdoctoral Mandate (PDM), grant number PDMT2/21/066.
TC is a PhD Fellow of the Research Foundation - Flanders (FWO), grant number 1166722N.
LD acknowledges support from KU Leuven C1 MAESTRO grant C16/17/007, KU Leuven C1 BRAVE grant C16/23/009, KU Leuven Methusalem grant METH24/012, and FWO Research grant G099720N.
TD is supported in part by the Australian Research Council through a Discovery Early Career Researcher Award, number DE230100183, and by the Australian Research Council Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D), through project number CE170100013.