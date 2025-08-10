Geodesic routines (pure Rust)
=============================

This crate provides a Rust port of the GeographicLib geodesic algorithms
needed by proj4rs. It previously bound to a C implementation, but now
implements the core direct and inverse solutions in Rust with equivalent
numerical behavior.

This crate is used as an optional dependency of the [proj4rs](https://lib.rs/crates/proj4rs)
and was created initially for the support of the [Azimuthal Equidistant (aeqd)](https://proj.org/en/stable/operations/projections/aeqd.html) projection.

