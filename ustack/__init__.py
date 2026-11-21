"""UStack: design van der Waals heterostructure devices and their e-beam lithography patterns.

Stage 1 (before fabrication): build a layer stack (or start from a template), get it validated
against the lab's rules, get the list of flakes to look for, and lay the flakes out on their
optical images.

Stage 2 (before each EBL step): align the device images to the chip's bond-pad GDS, trace the
flakes and defects, generate the EBL pattern (Hall bar etch, contacts, gates ...), re-check it
with the measured values and write a GDS with KLayout, plus an updated specification.

The non-GUI modules (config, model, templates, validation, flakes, geometry, hallbar, imaging,
alignment, gds_io, stage2, assistant) import without Qt and are what the tests exercise.
"""

__version__ = "0.1.0"
