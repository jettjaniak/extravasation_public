from wrappers.simulation_engine import CellParameters, ReceptorParameters, SimulationSettings, Parameters


DEFAULT_REC_P = dict(n_r=10000, lambda_=20., sigma=2, gamma=0.75, k_on_0=1.2*10**5, k_r_0=0.01)
DEFAULT_CELL_P = dict(r_c=4.5, mu=0.01, a_h=5.*10**-21, chi=70., temp=310, l_c=10, kappa=1/8, rho=5,
                      eps=7.08*10**-19, lam_ss=2.5*10**-6, diff_rho=0.05)
DEFAULT_SIM_P = dict(series_max_n=50, lvl1_over_r=-0.9, lvl1_points=16, test96=False, debug=False)

DEBUG_SIM_P = DEFAULT_SIM_P.copy()
DEBUG_SIM_P['debug'] = True
DEBUG_REC_P = DEFAULT_REC_P.copy()
DEBUG_REC_P['n_r'] = 7

P_SEL_REC_P = dict(n_r=10000, lambda_=20., sigma=2, gamma=1, k_on_0=1.2*10**5, k_r_0=0.01)
E_SEL_REC_P = dict(n_r=10000, lambda_=20., sigma=2, gamma=1, k_on_0=1.2*10**5, k_r_0=0.01)
ICAM_REC_P = dict(n_r=10000, lambda_=20., sigma=2, gamma=1, k_on_0=1.2*10**5, k_r_0=0.01)
VCAM_REC_P = dict(n_r=10000, lambda_=20., sigma=2, gamma=1, k_on_0=1.2*10**5, k_r_0=0.01)

cell_p = CellParameters(**DEFAULT_CELL_P)
sim_p = SimulationSettings(**DEFAULT_SIM_P)
p_sel_rec_p = ReceptorParameters(**P_SEL_REC_P)
e_sel_rec_p = ReceptorParameters(**E_SEL_REC_P)

p = Parameters(cell_p, sim_p)
p.add_receptor(p_sel_rec_p)
p.add_receptor(e_sel_rec_p)