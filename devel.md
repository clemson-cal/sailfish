# Sailfish development plan

- [ ] Orbital evolution
- [ ] Time series
- [ ] Upsampling
- [ ] HIP / ROCm port
- [ ] Multi-GPU
- [ ] Physics
      - PLM runtime parameter
      - Sink prescription
      - Alpha visc
      - Lambda / bulk visc term
- [ ] Max signal speed mode: a_max
      - Predicted: setup defines a fixed a_max
      - Discovered: driver computes a_max on the grid
      - Problem may define fixed a_max, a_ceiling,
        or both. If define, computed signal speed
        will be at most a_ceiling.
- [ ] Energy-conserving mode
- [ ] Safety features
      - Density / pressure floors
      - Fallbacks: use a new parameter, fallback
        interval, in addition to the fallback stack
        size, to control the time between recording
        fallback states.
      - Fake mass: we don't need this
- [ ] Reductions (e.g. sum, min, max)
      - For orbital evolution, use a global array of
        integrated source terms; each thread writes
        source terms each RK step to that array.
        The array must be totaled to output orbital
        evolution time series, but only at the time
        series cadence.
      - For discovering the max signal speed a_max,
        we either (a) "do it right" and do the global
        reduction on the card every time step, or
        (b) "play games" and use a fixed a_max based
        on knowledge of the distribution over time of
        computed a_max values.
