/* lax42.cpp
Author: Huaiyu Duan (UNM)
Description: C++ implementation of the algorithm developed by Joshua Martin.
*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <tuple>
#include <limits>
#include <cmath>
#include <iostream>

const double MAXSTEPINC = 1.2; // maximum increasing factor for a new step, NOT USED
const double MINSTEPDEC = 0.8; // minimum decreasing factor for a new step, NOT USED

using nparray = // NumPy dense array
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>; 

using namespace pybind11::literals;

const double inf = std::numeric_limits<double>::infinity();

// class of the Lax42 algorithm
class Lax42 {
    nparray nu0; // polarization vector at t
    nparray nu1; // polarization vectors at t+dt using one step
    nparray nu2; // polarization vectors at t+dt using two one steps with half dt 
    double *nu0_ptr, *nu1_ptr, *nu2_ptr; // pointers to the above data
    std::vector<double> dPdt0; // dP/dt at current time
    std::vector<double> P_int, dPdt_int; // polarization vectors and dP/dt in intermediate steps
    const double dz; // interval between spatial points
    const int Nz, Nu; // number of spatial and angular bins
    nparray u; // angular mesh
    nparray g0; // sum(g0 * P) gives the total polarization vector
    std::vector<double> g1; // g0*u
    const double tol; // tolerance
    const double max_step; // maximum step size
    double t_cur; // current time
    double dt_cur; // current step size
    double dt_pre; // previous step size

    int total_steps, success_steps; // number of total steps and successful steps

    // // copy Pin to Pout
    // void copy(const double *Pin, double *Pout);

    // compute dP/dt = ∂P/∂t + u * ∂P/∂z = H x P
    void compute_dPdt(const double *P, double *dPdt);

    // compute P at t+dt/2 using P_old and dP/dt_old at t, save it to P_int
    void halfStep(const double *P_old, const double *dPdt_old, const double dt);

    // use P and dP/dt at t+dt/2 (P_int and dPdt_int) and P at t (P_old) to compute P at t+dt (P_new)
    void oneStep(const double *P_old, double *P_new, const double dt);

    // compute the error
    double error();
    
    // Richardson extrapolation to improve the accuracy of the result
    // void richex();

    // compute one quality step
    void qualityStep();

    public:
    // constructor
    Lax42(const double t, const nparray P, const nparray z, const nparray u, const nparray g0, 
        const double atol=1e-8, const double rtol=1e-8, const double max_step=inf, const double dt=inf);

    // evolve the system to a new time t
    auto integrate(const double t);

    // get current time
    double get_t() const { return t_cur; }

    // set current time, for debugging purpose only
    void set_t(double t) { t_cur = t; }

    // get total steps
    int get_total_steps() const { return total_steps; }

    // get successful steps
    int get_success_steps() const { return success_steps; }
};


// constructor
Lax42::Lax42(const double _t, const nparray _P, const nparray _z, const nparray _u, const nparray _g0, const double _atol, const double _rtol, const double _max_step, const double _dt) :
nu0(_P), nu1(_P.request().shape), nu2(_P.request().shape), dz(_z.at(1)-_z.at(0)), Nz(_z.request().size), Nu(_u.request().size), u(_u), g0(_g0), 
tol(fmin(_atol, _rtol)), max_step(fmin(fabs(dz)*2.5, _max_step)), t_cur(_t), dt_cur(fmin(max_step, _dt)), dt_pre(dt_cur), total_steps(0), success_steps(0) {
    // sanity check
    if (Nz < 5)
        throw std::runtime_error("Lax42: Number of z bins should be larger than 4.");
    if (g0.request().size != Nu)
        throw std::runtime_error("Lax42: g0 has a wrong shape.");
    if (nu0.request().size != Nz*Nu*3)
        throw std::runtime_error("Lax42: P has a wrong shape.");
    // allocate storage
    dPdt0.resize(Nz*Nu*3);
    dPdt_int.resize(Nz*Nu*3);
    P_int.resize(Nz*Nu*3);
    g1.resize(Nu);

    // initialize data
    nu0_ptr = static_cast<double *>(nu0.request().ptr); // pointer to nu0 data
    nu1_ptr = static_cast<double *>(nu1.request().ptr); // pointer to nu1 data
    nu2_ptr = static_cast<double *>(nu2.request().ptr); // pointer to nu2 data
    double *uu = static_cast<double *>(u.request().ptr); // pointer to u data
    double *gg0 = static_cast<double *>(g0.request().ptr); // pointer to g0 data
    for (int iu = 0; iu < Nu; ++iu)
        g1[iu] = gg0[iu] * uu[iu];
    compute_dPdt(static_cast<double *>(nu0.request().ptr), &(dPdt0[0]));
}

// evolve the system to a new time t
auto Lax42::integrate(const double t) {
    if (t <= t_cur)
        throw std::runtime_error("Lax42::integrate(): Input time point has been reached.");

    const double eps = fmax(fabs(t_cur), fabs(t)) * 1e-15; 
    while(t - t_cur > eps) {
        if (t_cur + dt_cur > t)
            dt_cur = t - t_cur;
        qualityStep();
        if (t_cur + dt_cur == t_cur)
            throw std::runtime_error("Lax42::integrate(): Step size underflowed.");
    }

    if (nu0_ptr == static_cast<double *>(nu0.request().ptr))
        return std::make_tuple(t, nu0, dt_pre);
    else
        return std::make_tuple(t, nu2, dt_pre);
}

// compute one quality step
inline void Lax42::qualityStep() {
    // perform one step using step size dt_cur
    halfStep(nu0_ptr, &(dPdt0[0]), dt_cur);
    compute_dPdt(&(P_int[0]), &(dPdt_int[0]));
    oneStep(nu0_ptr, nu1_ptr, dt_cur); 

    // perform two steps using step size dt_cur/2
    halfStep(nu0_ptr, &(dPdt0[0]), dt_cur*0.5);
    compute_dPdt(&(P_int[0]), &(dPdt_int[0]));
    oneStep(nu0_ptr, nu2_ptr, dt_cur*0.5); 
    compute_dPdt(nu2_ptr, &(dPdt_int[0]));
    halfStep(nu2_ptr, &(dPdt_int[0]), dt_cur*0.5);
    compute_dPdt(&(P_int[0]), &(dPdt_int[0]));
    oneStep(nu2_ptr, nu2_ptr, dt_cur*0.5); 

    // validate step
    double err = error(); // normalized error
    if (err <= 1.) { // step succeeded
        compute_dPdt(nu2_ptr, &(dPdt0[0]));
        // Richardson extrapolation
        // richex();
        // swap pointers
        double *tmp = nu0_ptr;
        nu0_ptr = nu2_ptr;
        nu2_ptr = tmp;
        // update current time
        t_cur += dt_cur;
        ++success_steps;
        dt_pre = dt_cur;
    }
    ++total_steps;

    // compute new step size
    double coe = pow(err, -1./3.) * 0.98; 
    // dt_cur *= fmin(MAXSTEPINC, fmax(MINSTEPDEC, coe));
    dt_cur *= coe;
    dt_cur = fmin(dt_cur, max_step);
    // std::cerr << dt_cur << std::endl;
}

// copy Pin to Pout
// inline void Lax42::copy(const double *Pin, double *Pout) {
//     #ifdef _OPENMP
//     #pragma omp parallel for 
//     #endif
//     for (int iz = 0; iz < Nz; ++iz) {
//         const double *Pi = Pin + iz*3*Nu;
//         double *Po = Pout + iz*3*Nu;
//         #ifdef _OPENMP
//         #pragma omp simd 
//         #endif
//         for (int iiu = 0; iiu < Nu*3; ++iiu) {
//             Po[iiu] = Pi[iiu];
//         }
//     }
// }

// compute dP/dt = ∂P/∂t + u * ∂P/∂z = H x P
void Lax42::compute_dPdt(const double *P, double *dPdt) {
    double *uu = static_cast<double *>(u.request().ptr); // pointer to u data
    double *gg0 = static_cast<double *>(g0.request().ptr); // pointer to g0 data
    double *gg1 = &(g1[0]); // pointer to g1 data
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for (int iz = 0; iz < Nz; ++iz) {
        // sum of polarization vectors
        double Ptot0[3], Ptot1[3];
        for (int ii = 0; ii < 3; ++ii) {
            const double *Pi = P + (ii + iz*3) * Nu;
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

        const double *P0 = P + iz*3*Nu, *P1 = P0 + Nu, *P2 = P1 + Nu;
        double *d0 = dPdt + iz*3*Nu, *d1 = d0 + Nu, *d2 = d1 + Nu;
        #ifdef _OPENMP
        #pragma omp simd 
        #endif
        for (int iu = 0; iu < Nu; ++iu) {
            // self-coupling Hamiltonian
            double H0 = Ptot0[0] - uu[iu]*Ptot1[0];
            double H1 = Ptot0[1] - uu[iu]*Ptot1[1];
            double H2 = Ptot0[2] - uu[iu]*Ptot1[2];

            d0[iu] = H1 * P2[iu] - H2 * P1[iu];
            d1[iu] = H2 * P0[iu] - H0 * P2[iu];
            d2[iu] = H0 * P1[iu] - H1 * P0[iu];
        }
    }
}

// compute P at t+dt/2 using P_old and dP/dt_old at t, save it to P_int
void Lax42::halfStep(const double *P_old, const double *dPdt_old, const double dt) {
    const double coe1 = 9./16., coe3 = 1./16.;
    const double coe2 = 27.*dt/(48.*dz), coe4 = dt/(48.*dz);
    const double coe5 = 9.*dt/32., coe6 = dt/32.; // some coefficients
    double *uu = static_cast<double *>(u.request().ptr); // pointer to u data
    using carr = const double (*)[Nu]; // const 2D array
    using arr = double (*)[Nu];
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for (int iz = 0; iz < Nz; ++iz) {
        carr P0 = reinterpret_cast<carr>(P_old + iz*3*Nu);
        carr Pm1 = reinterpret_cast<carr>(P_old + ((iz+Nz-1)%Nz)*3*Nu);
        carr Pp1 = reinterpret_cast<carr>(P_old + ((iz+1)%Nz)*3*Nu);
        carr Pp2 = reinterpret_cast<carr>(P_old + ((iz+2)%Nz)*3*Nu);
        carr d0 = reinterpret_cast<carr>(dPdt_old + iz*3*Nu);
        carr dm1 = reinterpret_cast<carr>(dPdt_old + ((iz+Nz-1)%Nz)*3*Nu);
        carr dp1 = reinterpret_cast<carr>(dPdt_old + ((iz+1)%Nz)*3*Nu);
        carr dp2 = reinterpret_cast<carr>(dPdt_old + ((iz+2)%Nz)*3*Nu);
        arr Po = reinterpret_cast<arr>(&(P_int[0]) + iz*3*Nu);
        for (int ii = 0; ii < 3; ++ii) {
            #ifdef _OPENMP
            #pragma omp simd 
            #endif
            for (int iu = 0; iu < Nu; ++iu) {
                Po[ii][iu] = (coe1 - coe2 * uu[iu]) * Pp1[ii][iu] 
                    + (coe1 + coe2 * uu[iu]) * P0[ii][iu]
                    - (coe3 - coe4 * uu[iu]) * Pp2[ii][iu]
                    - (coe3 + coe4 * uu[iu]) * Pm1[ii][iu]
                    + coe5 * (dp1[ii][iu] + d0[ii][iu]) 
                    - coe6 * (dp2[ii][iu] + dm1[ii][iu]);
            }
        }        
    }
}

// use P and dP/dt at t+dt/2 (P_int and dPdt_int) and P at t (P_old) to compute P at t+dt (P_new)
void Lax42::oneStep(const double *P_old, double *P_new, const double dt) {
    const double coe1 = 27.*dt/(24.*dz), coe2 = dt/(24.*dz);
    const double coe3 = 9.*dt/16., coe4 = dt/16.;
    double *uu = static_cast<double *>(u.request().ptr); // pointer to u data
    using carr = const double (*)[Nu]; // const 2D array
    using arr = double (*)[Nu];
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for (int iz = 0; iz < Nz; ++iz) {
        carr Pp1 = reinterpret_cast<carr>(&(P_int[0]) + iz*3*Nu);
        carr Pm1 = reinterpret_cast<carr>(&(P_int[0]) + ((iz+Nz-1)%Nz)*3*Nu);
        carr Pp3 = reinterpret_cast<carr>(&(P_int[0]) + ((iz+1)%Nz)*3*Nu);
        carr Pm3 = reinterpret_cast<carr>(&(P_int[0]) + ((iz+Nz-2)%Nz)*3*Nu);
        carr dp1 = reinterpret_cast<carr>(&(dPdt_int[0]) + iz*3*Nu);
        carr dm1 = reinterpret_cast<carr>(&(dPdt_int[0]) + ((iz+Nz-1)%Nz)*3*Nu);
        carr dp3 = reinterpret_cast<carr>(&(dPdt_int[0]) + ((iz+1)%Nz)*3*Nu);
        carr dm3 = reinterpret_cast<carr>(&(dPdt_int[0]) + ((iz+Nz-2)%Nz)*3*Nu);
        carr Po = reinterpret_cast<carr>(P_old + iz*3*Nu);
        arr Pn = reinterpret_cast<arr>(P_new + iz*3*Nu);
        for (int ii = 0; ii < 3; ++ii) {
            #ifdef _OPENMP
            #pragma omp simd 
            #endif
            for (int iu = 0; iu < Nu; ++iu) {
                Pn[ii][iu] = Po[ii][iu]
                    + coe1 * uu[iu] * (Pm1[ii][iu] - Pp1[ii][iu]) 
                    - coe2 * uu[iu] * (Pm3[ii][iu] - Pp3[ii][iu])
                    + coe3 * (dp1[ii][iu] + dm1[ii][iu])
                    - coe4 * (dp3[ii][iu] + dm3[ii][iu]);
            }
        }        
    }
}

inline double SQR(double x) {
    return x*x;
}

// compute the error
double Lax42::error() {
    const double coe1 = 4./3., coe2 = -1./3.; // some coefficients
    double err = 0.; // maximum error
    #ifdef _OPENMP
    #pragma omp parallel for reduction(max: err)
    #endif
    for (int iz = 0; iz < Nz; ++iz) {
        double *Px0 = nu1_ptr + iz*Nu*3, *Px1 = Px0 + Nu, *Px2 = Px1 + Nu;
        double *Py0 = nu2_ptr + iz*Nu*3, *Py1 = Py0 + Nu, *Py2 = Py1 + Nu;
        #ifdef _OPENMP
        #pragma omp simd reduction(max: err)
        #endif
        for (int iu = 0; iu < Nu; ++iu) {
            err = fmax(err, SQR(Px0[iu] - Py0[iu]) + SQR(Px1[iu] - Py1[iu]) + SQR(Px2[iu] - Py2[iu]));
        }

        // Richardson extrapolation
        #ifdef _OPENMP
        #pragma omp simd 
        #endif
        for (int iiu = 0; iiu < 3*Nu; ++iiu) {
            Py0[iiu] = coe1*Py0[iiu] + coe2*Px0[iiu]; 
        }
    }

    return sqrt(err)/tol;
}

// Richardson extrapolation to improve the accuracy of the result
// void Lax42::richex() {
//     const double coe1 = 4./3., coe2 = -1./3.; // some coefficients
//     #ifdef _OPENMP
//     #pragma omp parallel for
//     #endif
//     for (int iz = 0; iz < Nz; ++iz) {
//         double *Px0 = nu1_ptr + iz*Nu*3;
//         double *Py0 = nu2_ptr + iz*Nu*3;
//         // Richardson extrapolation
//         #ifdef _OPENMP
//         #pragma omp simd 
//         #endif
//         for (int iiu = 0; iiu < 3*Nu; ++iiu) {
//             Py0[iiu] = coe1*Py0[iiu] + coe2*Px0[iiu]; 
//         }
//     }
// }


PYBIND11_MODULE(lax42, m) {
    pybind11::class_<Lax42>(m, "Lax42")
        .def(pybind11::init<const double, const nparray, const nparray, const nparray, const nparray, const double, const double, const double, const double>(), "t"_a, "P"_a, "z"_a, "u"_a, "g0"_a, "atol"_a=1e-8, "rtol"_a=1e-8, "max_step"_a=inf, "dt"_a=inf)
        .def("integrate", &Lax42::integrate)
        .def_property_readonly("t", &Lax42::get_t)
        .def_property_readonly("total_steps", &Lax42::get_total_steps)
        .def_property_readonly("success_steps", &Lax42::get_success_steps);;
}