#include "commons.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>


void mpi_printf(unsigned short node_id, const char *fmt, ...)
{
  if(mympi::rank == node_id){
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
  }
}

void mpi_fprintf(unsigned short node_id, FILE *stream, const char *fmt, ...)
{
  if(mympi::rank == node_id){
    va_list args;
    va_start(args, fmt);
    vfprintf(stream, fmt, args);
    va_end(args);
  }
}



void edison_assign_params(int argc, char *argv[], param_t *p_param)
{
  bool is_default;
  char buffer[100];
  char key[100];
  char value[100];
  char file_name[255];
  FILE *fp_inputdeck;

  // parameters from arguments
  for (int idx = 1; idx < argc; idx += 2) {
    if (!strcmp(argv[idx], "-input_deck"))
      strcpy(file_name, argv[idx + 1]);
    else if (!strcmp(argv[idx], "-hill_file"))
      strcpy(p_param->hill_file, argv[idx + 1]);
  }  

  is_default = false;
  fp_inputdeck = fopen( file_name, "r");
  if(fp_inputdeck == NULL){
    fprintf(stderr, "Cannot open input deck file %s!!!\nUse default value as the failsafe.\n", file_name);
    is_default = true;
  }

  // read input_deck line by line
  // and store each line to the buffer
  while ( is_default == false && fgets( buffer, 100, fp_inputdeck ) != NULL ) {
    sscanf( buffer, "%s %*s %s", key, value );
    if (strcmp(key, "Celltype") == 0) {
      p_param->celltype = strtod( value, NULL );
    }
    else if (strcmp(key, "Is_Dutta") == 0) {
      p_param->is_dutta = strtol( value, NULL, 10 );
    }
    else if (strcmp(key, "Is_Using_Output") == 0) {
      p_param->is_using_output = strtol( value, NULL, 10 );
    }
    else if (strcmp(key, "Basic_Cycle_Length") == 0) {
      p_param->bcl = strtod( value, NULL );
    }
    else if (strcmp(key, "Number_of_Pacing") == 0) {
      p_param->pace_max = strtol( value, NULL, 10 );
    }
    else if (strcmp(key, "Time_Step") == 0) {
      p_param->dt = strtod( value, NULL );
    }
    else if (strcmp(key, "Write_Step") == 0) {
      p_param->dt_write = strtod( value, NULL );
    }
    else if (strcmp(key, "Drug_Name") == 0) {
      strcpy( p_param->drug_name, value );
    }
    else if (strcmp(key, "Inet_Vm_Threshold") == 0) {
      p_param->inet_vm_threshold = strtod( value, NULL );
    }
    else if (strcmp(key, "Concentrations") == 0) {
      strcpy( p_param->concs, "0," );
      strcat( p_param->concs, value );
    }

  }

  if( is_default == false ) fclose( fp_inputdeck );
}
