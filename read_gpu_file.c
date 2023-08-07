#include <stdio.h>

struct sim_result{
  int core;
  float dt_set;
  float tcurr;
  float states;
  float rates;
  float GKs;
} ;

int main() {
  FILE *fp;
  struct sim_result result[100];
  int i;

  fp = fopen("testfile_2.dat", "rb");
  if (fp == NULL) {
    printf("Error opening file.\n");
    return 1;
  }

  for (i = 0; i < 100; i++) {
    fread(&result[i], sizeof(struct sim_result), 1, fp);
  }

  fclose(fp);
  printf("core,dt_set,tcurr,states,rates,GKs\n");
  for (i = 0; i < 100; i++) {
    printf("%d,%lf,%lf,%lf,%lf,%lf\n", result->core, result->dt_set, result->tcurr, result->states,result->rates, result->GKs);
  }

  return 0;
}