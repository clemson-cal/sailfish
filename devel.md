# Sailfish development plan

- [ ] Orbital evolution
- [ ] Time series
- [x] Upsampling
- [x] HIP / ROCm port
- [ ] Multi-GPU
- [ ] Physics
      - [ ] PLM runtime parameter
      - [x] Sink prescription
      - [ ] Alpha visc
      - [ ] Lambda / bulk visc term
- [x] Max signal speed mode: a_max
- [ ] Energy-conserving mode
- [ ] Safety features
      - [ ] Density / pressure floors
      - [ ] Fallbacks: use a new parameter, fallback interval, in addition to the
        fallback stack size, to control the time between recording fallback
        states.
      - [ ] Fake mass: we don't need this
- [ ] Reductions (e.g. sum, min, max)
      - [ ] For orbital evolution, use a global array of integrated source terms;
        each thread writes source terms each RK step to that array. The array
        must be totaled to output orbital evolution time series, but only at
        the time series cadence.
      - [x] For discovering the max signal speed a_max, we either (a) "do it
        right" and do the global reduction on the card every time step, or (b)
        "play games" and use a fixed a_max based on knowledge of the
        distribution over time of computed a_max values.
- [ ] Workflow improvements
      - Store command line flags in the checkpoint so they are restored
      - Give the problem an optional end time so if none is given the code
        doesn't just run forever
      + Add an option to only recompute timestep only at the fold boundaries
        because the time step
