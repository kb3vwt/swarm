/*************************************************************************
 * Copyright (C) 2010 by Mario Juric  and the Swarm-NG Development Team  *
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

/*! \file writer.h
 *   \brief Declares writer class
 *
*/
#pragma once

/// The main namespace for the Swarm-NG library
namespace swarm {

/**
        \brief Abstract output writer interface

        The method process() is called whenever the GPU output buffers are filled,
        and should proces/store the accumulated output data and logs (usually by
        writing them out to a file).
*/
class writer
{
        public:
                virtual void process(const char *log_data, size_t length) = 0;
                virtual ~writer() {};   // has to be here to ensure the derived class' destructor is called (if it exists)

        public:
                /// Integrator factory functions (and supporting typedefs)
                static writer *create(const std::string &cfg);

        protected:
                writer() {};            //!< hide the constructor.and force integrator instantiation with integrator::create
};
typedef writer *(*writerFactory_t)(const std::string &cfg);

}
