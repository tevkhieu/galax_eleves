#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <iostream>
namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;
Particles parts2(10000);

Model_CPU_fast ::Model_CPU_fast(const Initstate& initstate, Particles& particles) : Model_CPU(initstate, particles) {

    this->array_A = &particles;
    this->array_B = &parts2;


}

void Model_CPU_fast::step() {

// OMP + xsimd version
#pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&this->array_A->x[i]);
        const b_type rposy_i = b_type::load_unaligned(&this->array_A->y[i]);
        const b_type rposz_i = b_type::load_unaligned(&this->array_A->z[i]);
              b_type raccx_i = b_type(0.0);
              b_type raccy_i = b_type(0.0);
              b_type raccz_i = b_type(0.0);
              
        for (int j = 0; j < n_particles; j += b_type::size) {
            b_type rposx_j = b_type::load_unaligned(&this->array_A->x[j]);
            b_type rposy_j = b_type::load_unaligned(&this->array_A->y[j]);
            b_type rposz_j = b_type::load_unaligned(&this->array_A->z[j]);
            b_type rinitstatemasses = b_type::load_unaligned(&initstate.masses[j]);
            
            for (int k = 0; k < b_type::size; k++) {
                const b_type rdiffx = rposx_j - rposx_i;
                const b_type rdiffy = rposy_j - rposy_i;
                const b_type rdiffz = rposz_j - rposz_i;
    
                b_type rdij = rdiffx * rdiffx + rdiffy * rdiffy + rdiffz * rdiffz;
    
                rdij = rsqrt(rdij*rdij*rdij);
                rdij = min(10.0 * rdij, b_type(10.0));
                rdij = rdij * rinitstatemasses;

                raccx_i = xs::fma(rdiffx, rdij, raccx_i);
                raccy_i = xs::fma(rdiffy, rdij, raccy_i);
                raccz_i = xs::fma(rdiffz, rdij, raccz_i);
                
                rposx_j = xs::rotate_left<1>(rposx_j);
                rposy_j = xs::rotate_left<1>(rposy_j);
                rposz_j = xs::rotate_left<1>(rposz_j);
                rinitstatemasses = xs::rotate_left<1>(rinitstatemasses);
            }

        }

        //raccx_i.store_unaligned(&accelerationsx[i]);
        //raccy_i.store_unaligned(&accelerationsy[i]);
        //raccz_i.store_unaligned(&accelerationsz[i]);
        auto vec_x = b_type::load_unaligned(&velocitiesx[i]);
        auto vec_y = b_type::load_unaligned(&velocitiesy[i]);
        auto vec_z = b_type::load_unaligned(&velocitiesz[i]);

        vec_x += raccx_i * b_type(2.0f);
        vec_y += raccy_i * b_type(2.0f);
        vec_z += raccz_i * b_type(2.0f);

        vec_x.store_unaligned(&velocitiesx[i]);
        vec_y.store_unaligned(&velocitiesy[i]);
        vec_z.store_unaligned(&velocitiesz[i]);

        auto posx = b_type::load_unaligned(&this->array_B->x[i]);
        auto posy = b_type::load_unaligned(&this->array_B->y[i]);
        auto posz = b_type::load_unaligned(&this->array_B->z[i]);

        posx = rposx_i + vec_x*0.1f;
        posy = rposy_i + vec_y*0.1f;
        posz = rposz_i + vec_z*0.1f;

        posx.store_unaligned(&this->array_B->x[i]);
        posy.store_unaligned(&this->array_B->y[i]);
        posz.store_unaligned(&this->array_B->z[i]);

        //this->array_B->x[i] = this->array_A->x[i] + velocitiesx   [i] * 0.1f;
        //this->array_B->y[i] = this->array_A->y[i] + velocitiesy   [i] * 0.1f;
        //this->array_B->z[i] = this->array_A->z[i] + velocitiesz   [i] * 0.1f;
        
    }


    Particles* array_C = this->array_A;
    this->array_A = array_B;
    this->array_B = array_C;


}

#endif // GALAX_MODEL_CPU_FAST
