---
title: 'Bayesian model reconstruction based on spectral line observations with pomme'

tags:
  - Python
  - astronomy
  - radiative transfer

authors:
  - firstname: Frederik
    surname: De Ceuster^[Corresponding author.]
    orcid: 0000-0001-5887-8498
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Thomas Ceulemans
    orcid: 0000-0002-7808-9039
    affiliation: "1"
  - name: Leen Decin
    orcid: 0000-0002-5342-8612
    affiliation: "1"
  - name: Taissa Danilovich
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
This is especially the case for spectral line radiation caused by transitions between the quantized energy levels of atoms and molecules.
Rotational transition lines are particularly interesting, since they are excited in many astrophysical environments and can easily be resolved individually.
Using spectral line observations, we can infer information about physical and chemical parameters, such as abundance of certain molecules, velocity, and temperature distributions.
To facilitate this, we built a Python package, called [pomme]{.sc}, that helps users reconstruct 1D spherically symmetric or generic 3D models, based on astronomical spectral line observations.
A detailed description and validation of the methods can be found in [@DeCeuster2024].


# Statement of need
Spectral line observations are indispensible in astronomy.
As a result, many line radiation transport solvers exist to solve the (forward) problem of determining what spectral line observations of a certain model would look like [@DeCeuster2022; @Matsumoto2023].
However, far fewer tools exist to efficiently solve the more pressing inverse problem of determining what model could correspond to certain observations, commonly referred to as fitting our models to observtions.
Although one can use existing forward solvers to iteratively solve the inverse problem, the resulting process is often slow, cumbersome, and leaves, much room for improvement [@Coenegrachts2023; @Danilovich2024].
Therefore, in [pomme]{.sc}, we have implemented a line radiative transfer solver (the forward problem) in the [PyTorch]{.sc} framework [@Paszke2019] to leverage the [autograd]{.sc} functionality [@Paszke2017] to efficiently fit our models to the observations (the inverse problem) in a streamlined way [@DeCeuster2024].


# Example
\autoref{fig:example} shows an application of [pomme]{.sc}, where it was used to reconstruct the abundance distribution of NaCl (table salt) around the evolved star IK Tauri.
The reconstruction is based on observations of the NaCl ($J=26-25$) rotational line, taken with the Atacama Large (sub)Millimetre Array (ALMA), shown in \autoref{fig:example_obs}.
The original analysis was done by @Coenegrachts2023, and we improved their methods using [pomme]{.sc} as described in @DeCeuster2024.

![Reconstruction of the NaCl abundance distribution around the evolved star IK Tauri, created with [pomme]{.sc}. An interactive version of the figure is available in the [documentation](https://pomme.readthedocs.io/en/latest/_static/NaCl_reconstruction.html). \label{fig:example}](IKTau_NaCl.png)

![NaCl ($J=26-25$) rotational line observations, taken with the Atacama Large (sub)Millimetre Array (ALMA), which is used as input in [pomme]{.sc} to create a reconstruction. \label{fig:example_obs}](IKTau_NaCl_obs.png)


# Acknowledgements
FDC is a Postdoctoral Research Fellow of the Research Foundation - Flanders (FWO), grant number 1253223N, and was previously supported for this research by a KU Leuven Postdoctoral Mandate (PDM), grant number PDMT2/21/066.
TC is a PhD Fellow of the Research Foundation - Flanders (FWO), grant number 1166722N.
LD acknowledges support from KU Leuven C1 MAESTRO grant C16/17/007, KU Leuven C1 BRAVE grant C16/23/009, KU Leuven Methusalem grant METH24/012, and FWO Research grant G099720N.
TD is supported in part by the Australian Research Council through a Discovery Early Career Researcher Award, number DE230100183, and by the Australian Research Council Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D), through project number CE170100013.


# References