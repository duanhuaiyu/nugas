/* pdz.hpp
Author: Huaiyu Duan (UNM)
Description: Approximated first order derivatives on a periodic box using the central finite difference. 
*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <string>

using nparray = // NumPy dense array
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>; 

namespace ctrdiff {
    /* arguments of the functions:
    Nx : number of spatial points
    Ny : number of variables at one point
    double *y_tpr : array of function values, mapping to double[Nx][Ny]
    double *dydx_tpr : array of function values, mapping to double[Nx][Ny]
    */

    // 3-point central differencing
    void d3p(const double *y_ptr, double *dydx_ptr, const double dx, const int Nx, const int Ny) {
        using arr = double (*)[Ny]; // arr[Nx][Ny]
        using carr = const double (*)[Ny]; 
        carr y = reinterpret_cast<carr>(y_ptr);
        arr dydx = reinterpret_cast<arr>(dydx_ptr);
        const double coe = 0.5 / dx;
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for (int i = 0; i < Nx; ++i) {
            int m1 = (i+Nx-1)%Nx;
            int p1 = (i+1)%Nx;
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int j = 0; j < Ny; ++j)
                dydx[i][j] = coe * (-y[m1][j] + y[p1][j]);
        }
    }

    // 5-point central differencing
    void d5p(const double *y_ptr, double *dydx_ptr, const double dx, const int Nx, const int Ny) {
        using arr = double (*)[Ny]; // arr[Nx][Ny]
        using carr = const double (*)[Ny]; 
        carr y = reinterpret_cast<carr>(y_ptr);
        arr dydx = reinterpret_cast<arr>(dydx_ptr);
        const double coe = 1. / (12. * dx);
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for (int i = 0; i < Nx; ++i) {
            int m1 = (i+Nx-1)%Nx, m2 = (i+Nx-2)%Nx;
            int p1 = (i+1)%Nx, p2 = (i+2)%Nx;
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int j = 0; j < Ny; ++j)
                dydx[i][j] = coe * (y[m2][j] - 8.*y[m1][j] + 8.*y[p1][j] - y[p2][j]);
        }
    }

    // 7-point central differencing
    void d7p(const double *y_ptr, double *dydx_ptr, const double dx, const int Nx, const int Ny) {
        using arr = double (*)[Ny]; // arr[Nx][Ny]
        using carr = const double (*)[Ny]; 
        carr y = reinterpret_cast<carr>(y_ptr);
        arr dydx = reinterpret_cast<arr>(dydx_ptr);
        const double coe = 1. / (60. * dx);
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for (int i = 0; i < Nx; ++i) {
            int m1 = (i+Nx-1)%Nx, m2 = (i+Nx-2)%Nx, m3 = (i+Nx-3)%Nx;
            int p1 = (i+1)%Nx, p2 = (i+2)%Nx, p3 = (i+3)%Nx;
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int j = 0; j < Ny; ++j)
                dydx[i][j] = coe * (-y[m3][j] + 9.*y[m2][j] - 45.*y[m1][j] 
                    + 45.*y[p1][j] - 9.*y[p2][j] + y[p3][j]);
        }
    }

    // 9-point central differencing
    void d9p(const double *y_ptr, double *dydx_ptr, const double dx, const int Nx, const int Ny) {
        using arr = double (*)[Ny]; // arr[Nx][Ny]
        using carr = const double (*)[Ny]; 
        carr y = reinterpret_cast<carr>(y_ptr);
        arr dydx = reinterpret_cast<arr>(dydx_ptr);
        const double coe = 1. / (840. * dx);
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for (int i = 0; i < Nx; ++i) {
            int m1 = (i+Nx-1)%Nx, m2 = (i+Nx-2)%Nx, m3 = (i+Nx-3)%Nx, m4 = (i+Nx-4)%Nx;
            int p1 = (i+1)%Nx, p2 = (i+2)%Nx, p3 = (i+3)%Nx, p4 = (i+4)%Nx;
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int j = 0; j < Ny; ++j)
                dydx[i][j] = coe * (3.*y[m4][j] - 32.*y[m3][j] + 168.*y[m2][j] - 672.*y[m1][j] 
                    + 672.*y[p1][j] - 168.*y[p2][j] + 32.*y[p3][j] - 3.*y[p4][j]);
        }
    }

    
    // generator for the differentiator
    auto pdz(nparray x, const std::string &method = "fd5") {
        auto xinfo = x.request(); // array information
        const int Nx = xinfo.size; // number mesh points
        decltype(&d3p) dnp; // pointer to the derivative function
        if (method == "fd3") {
            if (Nx < 3)
                throw std::runtime_error("The number of mesh points must be larger than 2.");
            else
                dnp = d3p;
            }
        else if (method == "fd5") {
            if (Nx < 5)
                throw std::runtime_error("The number of mesh points must be larger than 4.");
            else
                dnp = d5p;
            }
        else if (method == "fd7") {
            if (Nx < 7)
                throw std::runtime_error("The number of mesh points must be larger than 6.");
            else
                dnp = d7p;
            }
        else if (method == "fd9") {
            if (Nx < 9)
                throw std::runtime_error("The number of mesh points must be larger than 8.");
            else
                dnp = d9p;
            }
        else
            throw std::runtime_error("Unknown differentiation method " + method + ".");      
        
        double *xx = static_cast<double*>(xinfo.ptr); // actual array
        double dx = xx[1] - xx[0]; // mesh interval
          
        return [=](nparray y) { // derivative function
            auto yinfo = y.request(); // array info
            if (yinfo.size % Nx)
                throw std::runtime_error("The shapes of y and x mismatch.");
            const int Ny = yinfo.size / Nx; // number of y elements on one x point
            nparray dydx(yinfo.shape); // NumPy array to store dydx
            double *yy = static_cast<double*>(yinfo.ptr); // actual array
            double *dyy = static_cast<double*>(dydx.request().ptr); 
            dnp(yy, dyy, dx, Nx, Ny);
            return dydx;
        };

    }

}
