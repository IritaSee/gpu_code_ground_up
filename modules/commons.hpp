#ifndef COMMONS_HPP
#define COMMONS_HPP

#include <cstdio>

#include "globals.hpp"
#include "param.hpp"
// #include "../cellmodel.hpp"

// custom printf for MPI
// to avoid duplicate printing
void mpi_printf(unsigned short node_id, const char *fmt, ...);
void mpi_fprintf(unsigned short node_id, FILE *stream, const char *fmt, ...);


// parameter setup function
void edison_assign_params(int argc, char *argv[], param_t *p_param);

#endif
