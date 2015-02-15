/*! \file closebinary.hpp
 *   \brief Defines \ref swarm::gpu::bppt::CloseBinaryPropagator - the GPU implementation 
 *          of John Chamber's Close Binary Propagator.
 *
 */

#include "swarm/common.hpp"
#include "swarm/swarmplugin.h"
#include "keplerian.hpp"

namespace swarm {

namespace gpu {
namespace bppt {

/*! Paramaters for CloseBinaryPropagator
 * \ingroup propagator_parameters
 *
 */
struct CloseBinaryPropagatorParams {
	double time_step;
        //! Constructor for CloseBinaryPropagatorParams
	CloseBinaryPropagatorParams(const config& cfg){
		time_step = cfg.require("time_step", 0.0);
	}
};

/*! GPU implementation of John Chamber's Close Binary propagator
 * \ingroup propagators
 *
 * 
 */
template<class T,class Gravitation>
struct CloseBinaryPropagator {
	typedef CloseBinaryPropagatorParams params;
	static const int nbod = T::n;

	params _params;


	//! Runtime variables
	ensemble::SystemRef& sys;
	Gravitation& calcForces;
	int b;
	int c;
	int ij;

	double sqrtGM;
	double max_timestep;

	double acc_bc;

        //! Constructor for CloseBinaryPropagator
	GPUAPI CloseBinaryPropagator(const params& p,ensemble::SystemRef& s,
			Gravitation& calc)
		:_params(p),sys(s),calcForces(calc){}

	__device__ bool is_in_body_component_grid()
        { return  ((b < nbod) && (c < 3)); }	

	__device__ bool is_in_body_component_grid_no_star()
        { return ( (b!=0) && (b < nbod) && (c < 3) ); }	

	__device__ bool is_first_thread_in_system()
        { return (thread_in_system()==0); }	

	static GENERIC int thread_per_system(){
		return nbod * 3;
	}

	static GENERIC int shmem_per_system() {
		 return 0;
	}



	/// Shift into Jacobi coordinate system 
	/// Initialization tasks executed before entering loop
        /// Cache sqrtGM, shift coord system, cache acceleration data for this thread's body and component
	GPUAPI void init()  { 
		sqrtGM = sqrt(sys[0].mass());
		convert_std_to_jacobi_coord_without_shared();
		__syncthreads();
		acc_bc = calcForces.acc_planets(ij,b,c);
                }

	/// Before exiting, convert back to standard cartesian coordinate system
	GPUAPI void shutdown() { 
	convert_jacobi_to_std_coord_without_shared();
	}

	///Convert to Jacobi Coordinates from Cartesian
	GPUAPI void convert_std_to_jacobi_coord_without_shared()  { 
	    double stdcoord_A,stdcoord_B;
		double sum_masspos = 0., sum_mom = 0., mtot = 0.;
		double mA = 0., mB = 0., nuA = 0., nuB = 0., momA = 0., momB = 0.;
		double jacobipos[nbod] = 0., jacobimom[nbod] = 0.;
		if( is_in_body_component_grid() )
		{
			//convert Binary Element A's position over to Jacobi
			stdcoord_A = sys[0][c].pos(); //Star A's cartesian coord
			stdcoord_B = sys[1][c].pos(); //Star B's cartesian coord
			mA = sys[0].mass(); //Star A's mass
			mB = sys[1].mass() //Star B's mass
			nuA = mA/(mA+mB); //mass fraction of A
			nuB = mB/(mA+mB); //mass fraction of B
			momA = mA * sys[0][c].vel(); //momentum of Star A
			momB = mB * sys[1][c].vel(); //momentum of Star B
			
			//Sum both the mass*pos, mass*vel (momentum) of each planet, total mass of planets, 
			for(int j=2;j<nbod;j++)
			{
				const double mj = sys[j].mass();
				mtot += mj;
				
				sum_masspos += mj*sys[j][c].pos();
				sum_mom += mj*sys[j][c].vel();
			}
			mtot += mA + mB; //add in star mass to total mass
			
			//calculate jacobi position of star A, B and the planet:
			jacobipos[0] = (sum_masspos + stdcoord_A*mA + stdcoord_B*mB)/mtot;
			jacobipos[1] = stdcoord_B - stdcoord_A;
			jacobipos[b] = sys[b][c].pos() - (nuA*stdcoord_A + nuB*stdcoord_B);
			
			//calculate jacobi/conjugate momenta of the stars A and B, and the planet's:
			jacobimom[0] = momA + momB + sum_mom;
			jacobimom[1] = momB - nuB * (momA - momB);
			jacobimom[b] = sys[b].mass() * sys[b][c].vel() - sys[b].mass()*(momA + momB + sum_mom)/mtot;
			
			}
		}
		__syncthreads();

		if( is_in_body_component_grid() )
		{
			sys[b][c].pos() = jacobipos[b]; //Finally switch to jacobi coordinates.
			sys[b][c].vel() = jacobimom / sys[b].mass(); // See note below:
			
			//Joe Sheldon: I propose a change to ensemble.hpp: save momenta within sys[][].mom()?
			//In the mean time, we'll divide out the mass of the respective body and store the velocity.
			//Upon integration, we'll just go ahead and re-multiply. Alternatively we could hypothetically
			//store momentum in sys[][].vel() but I'd rather not confuse the variables/units.
		}

	}

	///Convert back to Cartesian, from Jacobi
	GPUAPI void convert_jacobi_to_std_coord_without_shared()  { 
		
		double JPos_A = sys[0][c].pos(),
			   JPos_B = sys[1][c].pos(),
			   mA = sys[0].mass(),
			   mB = sys[1].mass(),
			   mplan = 0.,
			   mtot = 0.,
			   nuA = mA / (mA + mB),
			   nuB = mB / (mA + mB),
			   sum_masspos = 0.,
			   momA = 0.,
			   momB =0.,
			   sum_mom = 0.;
		double CartCoord_A = 0.,
			   CartCoord_B = 0.,
			   CartCoord_planet = 0.,
			   StdMom_A = 0.,
			   StdMom_B = 0.,
			   StdMom_planet = 0.;
			   
		if( is_in_body_component_grid() )
		{
			//Calculate SUM(mj*Jj)
			for(int j = 2;j<nbod;j++)
			{
				sum_masspos += sys[j].mass()*sys[j][c].pos();
				mplan += sys[j].mass();
				sum_mom += sys[j].mass()*sys[j][c].vel();
			}
			mtot = mA + mB + mplan;
			
			//Calculate Cartesian Coordinates:
			CartCoord_A = (JPos_A*mtot - (mB +mplan*nuB)*JPos_B - sum_masspos) / (mA + mB + mplan*(nuA + nuB));
			CartCoord_B = JPos_B + CartCoord_A;
			CartCoord_planet = sys[b][c].pos() + nuA*CartCoord_A + nuB*CartCoord_B;
			
			//calculate Momenta in Cartesian Coords
			StdMom_A = (1 - nuB) * ((1-mplan/mtot)*sys[0][c].vel()*mA - sum_mom - (sys[1][c].vel()*mB)/(1-nuB));
			StdMom_B = (sys[1][c].vel()*mB + nuB*StdMom_A)/(1-nuB);
			StdMom_planet = sys[b][c].vel()*sys[b].mass() + (sys[b].mass/mtot)*sys[0][c].vel()*mA;
			
		}
		__syncthreads();

		if( is_in_body_component_grid() )
		{
			sys[0][c].pos() = CartCoord_A;
			sys[1][c].pos() = CartCoord_B;
			sys[b][c].pos() = CartCoord_planet;
			
			sys[0][c].vel() = StdMom_A / mA;
			sys[1][c].vel() = StdMom_B / mB;
			sys[b][c].vel() = StdMom_planet / sys[b].mass();
		}

	}

	/// Standardized member name to call convert_jacobi_to_std_coord_without_shared() 
	GPUAPI void convert_internal_to_std_coord() 
	{ convert_jacobi_to_std_coord_without_shared();	} 

	/// Standardized member name to call convert_std_to_jacobi_coord_without_shared()
        GPUAPI void convert_std_to_internal_coord() 
	{ convert_std_to_jacobi__coord_without_shared(); }


	/// Drift step for MVS integrator
	GPUAPI void drift_step(const double hby2) 
	{
	     
	}


	/// Advance system by one time unit
	GPUAPI void advance()
	{
	    
	}
};




}
}
}

