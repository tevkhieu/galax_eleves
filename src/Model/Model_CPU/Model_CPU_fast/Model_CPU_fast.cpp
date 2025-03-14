#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <iostream>
namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{

// OMP + xsimd version
#pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
              b_type raccx_i = b_type(0.0);
              b_type raccy_i = b_type(0.0);
              b_type raccz_i = b_type(0.0);
              
        // for (int j = 0; j < n_particles; j++) {
        //     b_type rposx_j = b_type(particles.x[j]);
        //     b_type rposy_j = b_type(particles.y[j]);
        //     b_type rposz_j = b_type(particles.z[j]);
        //     b_type rinitstatemasses = b_type(initstate.masses[j]);

        //     const b_type rdiffx = rposx_j - rposx_i;
        //     const b_type rdiffy = rposy_j - rposy_i;
        //     const b_type rdiffz = rposz_j - rposz_i;

        //     b_type rdij = rdiffx * rdiffx + rdiffy * rdiffy + rdiffz * rdiffz;

        //     rdij = rsqrt(rdij*rdij*rdij);
        //     rdij = min(10.0 * rdij, b_type(10.0));

        //     raccx_i = raccx_i + (rdiffx * rdij * rinitstatemasses);
        //     raccy_i = raccy_i + (rdiffy * rdij * rinitstatemasses);
        //     raccz_i = raccz_i + (rdiffz * rdij * rinitstatemasses);
        // }

        for (int j = 0; j < n_particles; j += b_type::size) {
            b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
            b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
            b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
            b_type rinitstatemasses = b_type::load_unaligned(&initstate.masses[j]);
            
            for (int k = 0; k < b_type::size; k++) {
                const b_type rdiffx = rposx_j - rposx_i;
                const b_type rdiffy = rposy_j - rposy_i;
                const b_type rdiffz = rposz_j - rposz_i;
    
                b_type rdij = rdiffx * rdiffx + rdiffy * rdiffy + rdiffz * rdiffz;
    
                rdij = rsqrt(rdij*rdij*rdij);
                rdij = min(10.0 * rdij, b_type(10.0));
    
                raccx_i = raccx_i + (rdiffx * rdij * rinitstatemasses);
                raccy_i = raccy_i + (rdiffy * rdij * rinitstatemasses);
                raccz_i = raccz_i + (rdiffz * rdij * rinitstatemasses);
                
                rposx_j = xs::rotate_left<1>(rposx_j);
                rposy_j = xs::rotate_left<1>(rposy_j);
                rposz_j = xs::rotate_left<1>(rposz_j);
                rinitstatemasses = xs::rotate_left<1>(rinitstatemasses);
            }

        }

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
        
    }

    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx   [i] * 0.1f;
        particles.y[i] += velocitiesy   [i] * 0.1f;
        particles.z[i] += velocitiesz   [i] * 0.1f;
    }


}

#endif // GALAX_MODEL_CPU_FAST
