from wrappers.simulation_engine import SimulationState
from params import p


# TODO: disable gravity
def simulate(max_dist=100, max_time=5, shear=0, initial_height=0.03, save_every=1000):
    dt = 0.1 / p.max_k_on_0
    n_steps = int(max_time / dt)
    ss = SimulationState(p, dt, initial_height)
    result = ss.simulate_with_history(n_steps, shear, max_dist=max_dist, save_every=save_every)
    return result
