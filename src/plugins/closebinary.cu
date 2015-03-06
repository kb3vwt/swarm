/*! \file closebinary.cu
 *   \brief Initializes the GPU version of the close binary  propagator plugins.
 *
 */

#include "propagators/closebinary.hpp"
#include "monitors/composites.hpp"
#include "monitors/stop_on_ejection.hpp"
#include "monitors/log_time_interval.hpp"
#include "swarm/gpu/gravitation_acc.hpp"

//! Declare device_log variable 
typedef gpulog::device_log L;
using namespace swarm::monitors;
using namespace swarm::gpu::bppt;
using swarm::integrator_plugin_initializer;

//! Initialize the integrator plugin for mvs propagator
integrator_plugin_initializer< generic< CloseBinaryPropagator, stop_on_ejection<L>, GravitationAcc > >
	closebinary_prop_plugin("closebinary"
			,"This is the integrator based on the close binary propagator");

//! Initialize the integrator plugin for the close binary propagator for close_encounter event
integrator_plugin_initializer< generic< CloseBinaryPropagator, stop_on_ejection_or_close_encounter<L>, GravitationAcc  > >
	mvs_prop_ce_plugin("mvs_close_encounter"
			,"This is the integrator based on mvs propagator, monitor stop_on_ejection_or_close_encounter");


