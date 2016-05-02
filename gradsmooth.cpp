/*
    GradSmooth: Point cloud smoothing via distance to measure
    gradient flows.

    Author: Patrick A. O'Neil

License:
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include <iostream>
#include <gflags/gflags.h>
#include <cnpy/cnpy.h>
#include "easylogging++.h"

// Command line args
DEFINE_double(step_size, 0.1, "Step size for gradient flow");

// Start EasyLoggingPP logger
INITIALIZE_EASYLOGGINGPP;

int main(int argc, char** argv)
{
    // Set up logging
    el::Configurations conf("/home/poneil/Math/GradSmooth/logging.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    LOG(INFO) << "Starting GradSmooth.";

    return 0;
}
