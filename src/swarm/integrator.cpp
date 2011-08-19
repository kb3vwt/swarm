/*************************************************************************
 * Copyright (C) 2008-2010 by Mario Juric & Swarm-NG Development Team    *
 *                                                                       *
 * This program is free software; you can redistribute it and/or modify  *
 * it under the terms of the GNU General Public License as published by  *
 * the Free Software Foundation; either version 3 of the License.        *
 *                                                                       *
 * This program is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 * GNU General Public License for more details.                          *
 *                                                                       *
 * You should have received a copy of the GNU General Public License     *
 * along with this program; if not, write to the                         *
 * Free Software Foundation, Inc.,                                       *
 * 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ************************************************************************/

/*! \file integrator.cpp
 *  \brief class integrator 
 *
*/

#include "common.hpp"
#include "integrator.hpp"
#include "log/logmanager.hpp"
#include "plugin.hpp"

namespace swarm {

	void integrator::set_log_manager(log::manager& l){
		_log = l.get_hostlog();
	}
	void gpu::integrator::set_log_manager(log::manager& l){
		set_log(l.get_gpulog());
	}

	integrator::integrator(const config &cfg){
		set_log_manager(log::manager::default_log());
	}

	gpu::integrator::integrator(const config &cfg)
		: Base(cfg), _hens(Base::_ens) {
		set_log_manager(log::manager::default_log());
	}

/*!
   \brief Integrator instantiation support

  @param[in] cfg configuration class
*/

/* Must use factory class to dynamically load integrator subclass
 * instead of using constructor. Done so that users can define their
 * own integrators that the swarm code does not need to know about at
 * compile time
 */
integrator *integrator::create(const config &cfg)
{
        std::auto_ptr<integrator> integ;

        if(!cfg.count("integrator")) ERROR("Integrator type has to be chosen with 'integrator=xxx' keyword");

        std::string name = cfg.at("integrator");
        std::string plugin_name = "integrator_" + name;

		try {
			integ.reset( (integrator*) plugin::instance(plugin_name,cfg) );
		}catch(plugin_not_found& e){
			ERROR("Integrator " + name + " not found.");
		}

        return integ.release();
}


}
