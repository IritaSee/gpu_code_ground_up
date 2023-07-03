#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <array>
#include <vector>


// global variable for MPI.
struct mympi
{
  static char host_name[255];
  static int host_name_len;
  static int rank;
  static int size;
};


// data structure for IC50
//typedef std::vector< std::array<double,14> > drug_t;
typedef double drug_t[2800];

#endif
