#ifndef PARAM_HPP
#define PARAM_HPP

struct param_t
{
  bool is_dutta; // TRUE if using Dutta scaling
  bool is_using_output; // TRUE if using last output file
  double bcl; // basic cycle length
  unsigned short pace_max; // maximum pace
  double celltype;  // cell types
  double dt;        // time step
  double dt_write;  // writing step
  double inet_vm_threshold; // Vm threshold for calculating inet
  char hill_file[1024];
  char drug_name[100];
  char concs[100];
  void init();
  void show_val();
};

#endif
