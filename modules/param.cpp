#include "param.hpp"

#include <cstdio>
#include "commons.hpp"

void param_t::init()
{
  is_dutta = false;
  is_using_output = false;
  bcl = 2000.;
  pace_max = 1000;
  celltype = 0.;
  dt = 0.1;
  dt_write = 1.0;
  inet_vm_threshold = -88.0;
  snprintf(hill_file, sizeof(hill_file), "%s", "./drugs/bepridil/IC50_samples10.csv");
  snprintf(drug_name, sizeof(drug_name), "%s", "bepridil");
  snprintf(concs, sizeof(concs), "%s", "0.0");
}

void param_t::show_val()
{
  mpi_printf( 0, "%s -- %s\n", "Hill File", hill_file );
  mpi_printf( 0, "%s -- %lf\n", "Celltype", celltype);
  mpi_printf( 0, "%s -- %d\n", "Is_Dutta", is_dutta );
  mpi_printf( 0, "%s -- %d\n", "Is_Using_Output", is_using_output );
  mpi_printf( 0, "%s -- %lf\n", "Basic_Cycle_Length", bcl);
  mpi_printf( 0, "%s -- %d\n", "Number_of_Pacing", pace_max);
  mpi_printf( 0, "%s -- %lf\n", "Time_Step", dt);
  mpi_printf( 0, "%s -- %lf\n", "Inet_Vm_Threshold", inet_vm_threshold);
  mpi_printf( 0, "%s -- %lf\n", "Writing_Step", dt_write);
  mpi_printf( 0, "%s -- %s\n", "Drug_Name", drug_name);
  mpi_printf( 0, "%s -- %s\n", "Concentrations", concs);
}
