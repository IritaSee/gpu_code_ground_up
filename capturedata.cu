#include "cuda_runtime.h"
#include <iostream>

#include "modules/globals.hpp"
#include "modules/commons.hpp"

char buffer[255];
drug_t ic50;
// __shared__ drug_t *d_ic50;
double *d_concs;

int get_IC50_data_from_file(const char* file_name, double *ic50)
{
  FILE *fp_drugs;
//   drug_t ic50;
  char *token;
  
  unsigned short idx;

  if( (fp_drugs = fopen(file_name, "r")) == NULL){
    printf("Cannot open file %s\n",
      file_name);
    return 0;
  }
  idx = 0;
  int sample_size = 0;
  fgets(buffer, sizeof(buffer), fp_drugs); // skip header
  while( fgets(buffer, sizeof(buffer), fp_drugs) != NULL )
  { // begin line reading
    token = strtok( buffer, "," );
    while( token != NULL )
    { // begin data tokenizing
      ic50[idx++] = strtod(token, NULL);
      token = strtok(NULL, ",");
    } // end data tokenizing
    sample_size++;
  } // end line reading

  fclose(fp_drugs);
  return sample_size;
}

__global__ void check(drug_t d_ic50, int sample_size){
 unsigned short sample_id;
    sample_id = threadIdx.x;
    { // begin sample loop
        printf("Sample_ID:%d \nData: ",
        sample_id );
        
        for (int z=0+(sample_id*14);z<(sample_id*14)+14;z++){
            printf("Core %d ic50[%d]: %lf \n",sample_id, z, d_ic50[z]);
        }
        // printf("\n");

    } // end sample loop
}

double concs[4];
int main()
{
    double *d_ic50;
    // input variables for cell simulation

    unsigned short idx;

    snprintf(buffer, sizeof(buffer),
      "./IC50_samples10.csv");
    int sample_size = get_IC50_data_from_file(buffer, ic50);
    // if(ic50.size() == 0)
    //     printf("Something problem with the IC50 file!\n");
    // else if(ic50.size() > 2000)
    //     printf("Too much input! Maximum sample data is 2000!\n");
    cudaMalloc(&d_ic50, sizeof(drug_t));
    cudaMemcpy(d_ic50, ic50, sizeof(drug_t), cudaMemcpyHostToDevice);
    check<<<1,10>>>(d_ic50, sample_size);
    cudaDeviceSynchronize();
    // unsigned short sample_id;
    // for( sample_id = 0;
    //     sample_id < sample_size;
    //     sample_id ++ )
    // { // begin sample loop
    //     printf("Sample_ID:%d \nData: ",
    //     sample_id );
        
    //     for (int z=0+(sample_id*14);z<(sample_id*14)+14;z++){
    //         printf("%lf ",ic50[z]);
    //     }
    //     printf("\n");

    // } // end sample loop
    
    return 0;
}
