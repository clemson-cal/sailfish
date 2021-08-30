# Sailfish development plan

- [x] Time series capability
- [ ] Orbital evolution
      - [x] Mass accretion and forces time series
      - [ ] Orbital evolution time series
- [x] Upsampling
- [x] HIP / ROCm port
- [x] Multi-GPU
- [ ] Multi-GPU + MPI
- [ ] Physics
      - [ ] PLM runtime parameter
      - [x] Sink prescription
      - [ ] Alpha visc
      - [x] Lambda / bulk visc term
- [x] Max signal speed mode: a_max
- [x] Energy-conserving mode
- [ ] Safety features
      - [ ] Density / pressure floors
      - [ ] Fallbacks: use a new parameter, fallback interval, in addition to the
        fallback stack size, to control the time between recording fallback
        states.
      - [ ] Fake mass: we don't need this
- [x] Reductions (e.g. sum, min, max)
      - [x] For orbital evolution, use a global array of integrated source
        terms; each thread writes source terms each RK step to that array. The
        array must be totaled to output orbital evolution time series, but
        only at the time series cadence. [WIP]
      - [x] For discovering the max signal speed a_max, we either (a) "do it
        right" and do the global reduction on the card every time step, or (b)
        "play games" and use a fixed a_max based on knowledge of the
        distribution over time of computed a_max values.
- [x] Workflow improvements
      + [x] Store command line flags in the checkpoint so they are restored
      + [x] Log-space checkpoint mode
      + [x] Give the problem an optional end time so if none is given the code
        doesn't just run forever
      + [x] Add an option to only recompute timestep only at the fold
        boundaries because the time step
