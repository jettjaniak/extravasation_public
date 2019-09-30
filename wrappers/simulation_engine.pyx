from SimulationState cimport SimulationState as SimulationStateCpp
from Parameters cimport (
    CellParameters as CellParametersCpp, SimulationSettings as SimulationSettingsCpp,
    ReceptorParameters as ReceptorParametersCpp, Parameters as ParametersCpp
)
from cpython cimport array
import numpy as np
cimport numpy as cnp
import array
cimport cython

from libc cimport limits


cdef class SimulationState:
    cdef SimulationStateCpp ss_cpp
    cdef object p

    def __cinit__(self, Parameters p, double dt_ = -1, double h_0 = -1, unsigned long int seed = 0):
        # When we call constructor with only first parameter, we don't create C++ class instance.
        if dt_ != -1 and h_0 != -1:
            if seed == 0:
                seed = np.random.randint(limits.ULONG_MAX, dtype='uint')
            self.ss_cpp = SimulationStateCpp(&p.p_cpp, dt_, h_0, seed)

    def __init__(self, Parameters p, double dt_ = -1, double h_0 = -1, unsigned long int seed = 0):
        self.p = p

    def copy(self, unsigned long int seed = 0):
        copied_ss = SimulationState(self.p)
        # Here the default C++ copy constructor is used.
        copied_ss.ss_cpp = self.ss_cpp
        if seed != 0:
            copied_ss.ss_cpp.reseed(seed)
        return copied_ss

    def __copy__(self):
        return self.copy(self)

    @cython.cdivision(True)
    def simulate_with_history(self, long int n_steps, double shear = 0., double normal = 0., double tangential = 0.,
                              bint stop_if_no_bonds = False, double max_dist=100, long int save_every=1000):
        cdef long int i = 0, j, hist_size, hist_i = 0
        hist_size = 1 + (n_steps - 1) // save_every

        cdef array.array h = array.array('d', [])
        cdef array.array dist = array.array('d', [])
        cdef array.array n_bonds = array.array('l', [])
        array.resize(h, hist_size)
        array.resize(dist, hist_size)
        array.resize(n_bonds, hist_size)

        cdef cnp.double_t[:, :] v = np.empty([hist_size, 6], dtype=np.double)
        cdef cnp.double_t[:, :] f = np.empty([hist_size, 6], dtype=np.double)

        for i in range(n_steps):
            if stop_if_no_bonds and self.ss_cpp.n_bonds == 0:
                print(f"No bonds in step {i}, stopping.")
                break
            elif self.ss_cpp.dist > max_dist:
                print(f"Traveled maximal distance in {i} steps, stopping.")
                break
            if i % save_every == 0:
                h[hist_i] = self.ss_cpp.h
                dist[hist_i] = self.ss_cpp.dist
                n_bonds[hist_i] = self.ss_cpp.n_bonds
                self.ss_cpp.simulate_one_step(shear, normal, tangential)

                v[hist_i][0] = self.ss_cpp.v.v_x
                v[hist_i][1] = self.ss_cpp.v.v_y
                v[hist_i][2] = self.ss_cpp.v.v_z
                v[hist_i][3] = self.ss_cpp.v.o_x
                v[hist_i][4] = self.ss_cpp.v.o_y
                v[hist_i][5] = self.ss_cpp.v.o_z

                f[hist_i][0] = self.ss_cpp.f.f_x
                f[hist_i][1] = self.ss_cpp.f.f_y
                f[hist_i][2] = self.ss_cpp.f.f_z
                f[hist_i][3] = self.ss_cpp.f.t_x
                f[hist_i][4] = self.ss_cpp.f.t_y
                f[hist_i][5] = self.ss_cpp.f.t_z

                hist_i += 1
            else:
                self.ss_cpp.simulate_one_step(shear, normal, tangential)

        return dict(
            h=h[:hist_i], dist=dist[:hist_i], n_bonds=n_bonds[:hist_i],
            v=np.asarray(v[:hist_i]), f=np.asarray(f[:hist_i])
        )

    def simulate_without_history(self, long int n_steps, double shear = 0., double normal = 0., double tangential = 0.,
                                 bint stop_if_no_bonds = False):
        for i in range(n_steps):
            if stop_if_no_bonds and self.ss_cpp.n_bonds == 0:
                # print(f"No bonds in step {i}, stopping.")
                break
            self.ss_cpp.simulate_one_step(shear, normal, tangential)

    # Attribute access
    @property
    def h(self):
        return self.ss_cpp.h
    @h.setter
    def h(self, h):
        self.ss_cpp.h = h

    @property
    def dist(self):
        return self.ss_cpp.dist

    @property
    def n_bonds(self):
        return self.ss_cpp.n_bonds

    @property
    def n_lvl1_updates(self):
        return self.ss_cpp.n_lvl1_updates


cdef class CellParameters:
    cdef CellParametersCpp cell_p_cpp

    def __init__(self, double r_c, double mu, double a_h, double chi, double temp,
		         double l_c, double kappa, double rho, double eps, double lam_ss, double diff_rho):
        self.cell_p_cpp = CellParametersCpp(r_c, mu, a_h, chi, temp,
		                                    l_c, kappa, rho, eps, lam_ss, diff_rho)

    @property
    def r_c(self):
        return self.cell_p_cpp.r_c
    @r_c.setter
    def r_c(self, r_c):
        self.cell_p_cpp.r_c = r_c

    @property
    def mu(self):
        return self.cell_p_cpp.mu
    @mu.setter
    def mu(self, mu):
        self.cell_p_cpp.mu = mu
        
    @property
    def a_h(self):
        return self.cell_p_cpp.a_h
    @a_h.setter
    def a_h(self, a_h):
        self.cell_p_cpp.a_h = a_h
        
    @property
    def chi(self):
        return self.cell_p_cpp.chi
    @chi.setter
    def chi(self, chi):
        self.cell_p_cpp.chi = chi
        
    @property
    def temp(self):
        return self.cell_p_cpp.temp
    @temp.setter
    def temp(self, temp):
        self.cell_p_cpp.temp = temp
        
    @property
    def l_c(self):
        return self.cell_p_cpp.l_c
    @l_c.setter
    def l_c(self, l_c):
        self.cell_p_cpp.l_c = l_c
        
    @property
    def kappa(self):
        return self.cell_p_cpp.kappa
    @kappa.setter
    def kappa(self, kappa):
        self.cell_p_cpp.kappa = kappa
        
    @property
    def rho(self):
        return self.cell_p_cpp.rho
    @rho.setter
    def rho(self, rho):
        self.cell_p_cpp.rho = rho
        
    @property
    def eps(self):
        return self.cell_p_cpp.eps
    @eps.setter
    def eps(self, eps):
        self.cell_p_cpp.eps = eps
        
    @property
    def lam_ss(self):
        return self.cell_p_cpp.lam_ss
    @lam_ss.setter
    def lam_ss(self, lam_ss):
        self.cell_p_cpp.lam_ss = lam_ss
        
    @property
    def diff_rho(self):
        return self.cell_p_cpp.diff_rho
    @diff_rho.setter
    def diff_rho(self, diff_rho):
        self.cell_p_cpp.diff_rho = diff_rho
        
    @property
    def grav_force(self):
        return self.cell_p_cpp.grav_force
    @grav_force.setter
    def grav_force(self, grav_force):
        # TODO: update if parameters changed
        self.cell_p_cpp.grav_force = grav_force


cdef class SimulationSettings:
    cdef SimulationSettingsCpp sim_p_cpp

    def __init__(self, int series_max_n, double lvl1_over_r,
		         int lvl1_points, bint test96, bint debug):
        self.sim_p_cpp = SimulationSettingsCpp(series_max_n, lvl1_over_r, lvl1_points, test96, debug)

    @property
    def series_max_n(self):
        return self.sim_p_cpp.series_max_n
    @series_max_n.setter
    def series_max_n(self, series_max_n):
        self.sim_p_cpp.series_max_n = series_max_n

    @property
    def lvl1_points(self):
        return self.sim_p_cpp.lvl1_points
    @lvl1_points.setter
    def lvl1_points(self, lvl1_points):
        self.sim_p_cpp.lvl1_points = lvl1_points

    @property
    def lvl1_over_r(self):
        return self.sim_p_cpp.lvl1_over_r
    @lvl1_over_r.setter
    def lvl1_over_r(self, lvl1_over_r):
        self.sim_p_cpp.lvl1_over_r = lvl1_over_r
        
    @property
    def test96(self):
        return self.sim_p_cpp.test96
    @test96.setter
    def test96(self, test96):
        self.sim_p_cpp.test96 = test96
        
    @property
    def debug(self):
        return self.sim_p_cpp.debug
    @debug.setter
    def debug(self, debug):
        self.sim_p_cpp.debug = debug


cdef class ReceptorParameters:
    cdef ReceptorParametersCpp rec_p_cpp

    def __init__(self, long int n_r, double lambda_, double sigma,
		         double gamma, double k_on_0, double k_r_0):
        self.rec_p_cpp = ReceptorParametersCpp(n_r, lambda_, sigma, gamma, k_on_0, k_r_0)
    
    @property
    def n_r(self):
        return self.rec_p_cpp.n_r
    @n_r.setter
    def n_r(self, n_r):
        self.rec_p_cpp.n_r = n_r
        
    @property
    def lambda_(self):
        return self.rec_p_cpp.lambda_
    @lambda_.setter
    def lambda_(self, lambda_):
        self.rec_p_cpp.lambda_ = lambda_
        
    @property
    def sigma(self):
        return self.rec_p_cpp.sigma
    @sigma.setter
    def sigma(self, sigma):
        self.rec_p_cpp.sigma = sigma
    
    @property
    def gamma(self):
        return self.rec_p_cpp.gamma
    @gamma.setter
    def gamma(self, gamma):
        self.rec_p_cpp.gamma = gamma
    
    @property
    def k_on_0(self):
        return self.rec_p_cpp.k_on_0
    @k_on_0.setter
    def k_on_0(self, k_on_0):
        self.rec_p_cpp.k_on_0 = k_on_0
        
    @property
    def k_r_0(self):
        return self.rec_p_cpp.k_r_0
    @k_r_0.setter
    def k_r_0(self, k_r_0):
        self.rec_p_cpp.k_r_0 = k_r_0


cdef class Parameters:
    cdef ParametersCpp p_cpp
    cdef object cell_p
    cdef object sim_p
    cdef object rec_p

    def __init__(self, CellParameters cell_p, SimulationSettings sim_p):
        self.cell_p = cell_p
        self.sim_p = sim_p
        self.rec_p = []
        self.p_cpp = ParametersCpp(&cell_p.cell_p_cpp, &sim_p.sim_p_cpp)

    def add_receptor(self, ReceptorParameters new_rec):
        self.rec_p.append(new_rec)
        self.p_cpp.add_receptor(&new_rec.rec_p_cpp)

    @property
    def n_rec_types(self):
        return self.p_cpp.n_rec_types

    @property
    def cell(self):
        return self.cell_p

    @property
    def sim(self):
        return self.sim_p

    @property
    def rec(self):
        return self.rec_p

    @property
    def max_k_on_0(self):
        return max([self.rec_p[i].k_on_0 for i in range(self.n_rec_types)])