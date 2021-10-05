/* eom_c.hpp
Author: Huaiyu Duan (UNM)
Description: C++ implementation of dyn1d that provides the time derivative of the polarization vectors in the dynamic 1D model.
*/
#include <string>
#include <vector>
#include "../misc/pdz.hpp"

using namespace pybind11::literals;

/* Generates a function that compute the time derivative of the polarization vector.
    z[Nz] : NumPy array of the spatial mesh points
    u[Nu] : NumPy array of the angular mesh points
    g0[Nu] : P dot g0 gives the integral of P over u
    Dz : method to compute the derivative along z.
    return : dPdt(t, P)
*/
auto eom(nparray z, nparray u, nparray g0, const std::string &Dz) {
    auto pdz = ctrdiff::pdz(z, Dz); // spatial derivative 
    const int Nz = z.request().size; // number of spatial points
    const int Nu = u.request().size; // number of angle bins
    double *uu = static_cast<double *>(u.request().ptr);
    double *gg0 = static_cast<double *>(g0.request().ptr);

    // compute g1
    std::vector<double> g1(Nu);
    for (int iu = 0; iu < Nu; ++iu)
        g1[iu] = gg0[iu] * uu[iu];

    return [=](const double t, nparray P) {
        const double *gg1 = &(g1[0]);
        auto Pinfo = P.request(); // array information
        const int Psize = Pinfo.size; // size of P
        if (Psize != Nz*Nu*3)
            throw std::runtime_error("P has a wrong shape.");
        double *PP = static_cast<double*>(P.request().ptr); // actual data

        nparray r = pdz(P); // ∂P/∂z
        double *rr = static_cast<double*>(r.request().ptr); // actual data

        // changing rate of the polarization vectors : ∂P/∂t = H x P - u ∂P/∂z 
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for (int iz = 0; iz < Nz; ++iz) {
            // sum of polarization vectors
            double Ptot0[3], Ptot1[3];
            for (int ii = 0; ii < 3; ++ii) {
                double *Pi = PP + (ii + iz*3) * Nu;
                double sum0 = 0., sum1 = 0.;
                #ifdef _OPENMP
                #pragma omp simd reduction(+: sum0, sum1)
                #endif
                for (int iu = 0; iu < Nu; ++iu) {
                    sum0 += Pi[iu] * gg0[iu];
                    sum1 += Pi[iu] * gg1[iu];
                }
                Ptot0[ii] = sum0;
                Ptot1[ii] = sum1;
            }

            double *P0 = PP + iz*3*Nu, *P1 = P0 + Nu, *P2 = P1 + Nu;
            double *r0 = rr + iz*3*Nu, *r1 = r0 + Nu, *r2 = r1 + Nu;
            #ifdef _OPENMP
            #pragma omp simd 
            #endif
            for (int iu = 0; iu < Nu; ++iu) {
                // self-coupling Hamiltonian
                double H0 = Ptot0[0] - uu[iu]*Ptot1[0];
                double H1 = Ptot0[1] - uu[iu]*Ptot1[1];
                double H2 = Ptot0[2] - uu[iu]*Ptot1[2];

                r0[iu] = H1 * P2[iu] - H2 * P1[iu] - uu[iu] * r0[iu];
                r1[iu] = H2 * P0[iu] - H0 * P2[iu] - uu[iu] * r1[iu];
                r2[iu] = H0 * P1[iu] - H1 * P0[iu] - uu[iu] * r2[iu];
            }
        }

        return r;
    };
}

// python wrapper for eom
pybind11::cpp_function eom_py(nparray z, nparray u, nparray g0, const std::string &Dz) {
    return pybind11::cpp_function(eom(z, u, g0, Dz));
}

// expose dPdt() to Python
PYBIND11_MODULE(eom_c, m) {
    m.def("eom", &eom_py, 
    "Generate a function that computes the time derivative of the polarization vectors.", 
    "z"_a, "u"_a, "g0"_a, "Dz"_a="fd5");
}