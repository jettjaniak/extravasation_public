from libcpp.vector cimport vector

cdef extern from "../ekstrawazacja_cpp/Parameters.h":
    cdef cppclass CellParameters:
        double r_c, mu, a_h, chi, temp, l_c, kappa, rho, eps, lam_ss, diff_rho, grav_force

        CellParameters() except +
        CellParameters(double r_c_, double mu_, double a_h_, double chi_, double temp_,
		    double l_c_, double kappa_, double rho_, double eps_, double lam_ss_, double diff_rho_)

    cdef cppclass SimulationSettings:
        int series_max_n, lvl1_points
        double lvl1_over_r
        bint test96, debug

        SimulationSettings() except +
        SimulationSettings(int series_max_n_, double lvl1_over_r_,
		    int lvl1_points_, bint test96_, bint debug_)

    cdef cppclass ReceptorParameters:
        long int n_r
        double lambda_, sigma, gamma, k_on_0, k_r_0

        ReceptorParameters() except +
        ReceptorParameters(long int n_r_, double lambda__, double sigma_,
		    double gamma_, double k_on_0_, double k_r_0_)

    cdef cppclass Parameters:
        CellParameters* cell
        SimulationSettings* sim
        vector[ReceptorParameters*] rec
        int n_rec_types

        Parameters() except +
        Parameters(CellParameters* cell_, SimulationSettings* sim_)
        void add_receptor(ReceptorParameters* new_rec)

