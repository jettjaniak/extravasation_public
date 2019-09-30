from Parameters cimport Parameters


cdef extern from "../ekstrawazacja_cpp/SimulationState.h":
    cdef cppclass velocities:
        double v_x, v_y, v_z, o_x, o_y, o_z

    cdef cppclass forces:
        double f_x, f_y, f_z, t_x, t_y, t_z

    cdef cppclass SimulationState:
        SimulationState() except +
        SimulationState(Parameters * p_, double dt_, double h_0, unsigned long int seed) except +
        Parameters * p
        velocities v
        forces f
        double h
        double h_diff
        double dist
        long int n_bonds, n_lvl1_updates
        void simulate_one_step(double shear, double normal, double tangential)
        void reseed(unsigned long int seed) except +