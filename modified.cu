#include "cuda_runtime.h"
#include <iostream>
#include <math.h>
#include <ctime>

#include "modules/globals.hpp"
#include "modules/commons.hpp"

#include "enums/enum_mar_cell_MKII.cuh"

#define ENOUGH ((CHAR_BIT * sizeof(int) - 1) / 3 + 2)

clock_t START_TIMER;

clock_t tic();
void toc(clock_t start = START_TIMER);

clock_t tic()
{
    return START_TIMER = clock();
}

void toc(clock_t start)
{
    std::cout
        << "Elapsed time: "
        << (clock() - start) / (double)CLOCKS_PER_SEC << "s"
        << std::endl;
}

// -----------------------------------------
__device__ void initConsts(double *CONSTANTS, double *STATES){
// int offset = threadIdx.x;
int offset = blockIdx.x * blockDim.x + threadIdx.x;
int num_of_constants = 146;
int num_of_states = 41;
// printf("init const for core %d\n",offset);
CONSTANTS[celltype+(offset * num_of_constants)] = 0;
CONSTANTS[R+(offset * num_of_constants)] = 8314;
CONSTANTS[T+(offset * num_of_constants)] = 310;
CONSTANTS[F+(offset * num_of_constants)] = 96485;
CONSTANTS[cm+(offset * num_of_constants)] = 1;
CONSTANTS[rad+(offset * num_of_constants)] = 0.0011;
CONSTANTS[L+(offset * num_of_constants)] = 0.01;
CONSTANTS[vcell+(offset * num_of_constants)] =  1000.00*3.14000*CONSTANTS[rad+(offset * num_of_constants)]*CONSTANTS[rad+(offset * num_of_constants)]*CONSTANTS[L+(offset * num_of_constants)];
CONSTANTS[amp+(offset * num_of_constants)] = -80;
CONSTANTS[duration+(offset * num_of_constants)] = 0.5;
CONSTANTS[zna+(offset * num_of_constants)] = 1;
CONSTANTS[zca+(offset * num_of_constants)] = 2;
CONSTANTS[zk+(offset * num_of_constants)] = 1;
CONSTANTS[stim_start+(offset * num_of_constants)] = 10.0;
CONSTANTS[stim_end+(offset * num_of_constants)] = 100000000000000000;
CONSTANTS[stim_period+(offset * num_of_constants)] = 1000.0;
CONSTANTS[step_low+(offset * num_of_constants)] = -150.;
CONSTANTS[step_high+(offset * num_of_constants)] = 0;
CONSTANTS[step_start+(offset * num_of_constants)] = 10;
CONSTANTS[step_end+(offset * num_of_constants)] = 5000;
CONSTANTS[GNa+(offset * num_of_constants)] = 75;
CONSTANTS[CaMKo+(offset * num_of_constants)] = 0.05;
CONSTANTS[KmCaM+(offset * num_of_constants)] = 0.0015;
CONSTANTS[KmCaMK+(offset * num_of_constants)] = 0.15;
CONSTANTS[nao+(offset * num_of_constants)] = 140;
CONSTANTS[mssV1+(offset * num_of_constants)] = 39.57;
CONSTANTS[mssV2+(offset * num_of_constants)] = 9.871;
CONSTANTS[mtD1+(offset * num_of_constants)] = 6.765;
CONSTANTS[mtD2+(offset * num_of_constants)] = 8.552;
CONSTANTS[mtV1+(offset * num_of_constants)] = 11.64;
CONSTANTS[mtV2+(offset * num_of_constants)] = 34.77;
CONSTANTS[mtV3+(offset * num_of_constants)] = 77.42;
CONSTANTS[mtV4+(offset * num_of_constants)] = 5.955;
CONSTANTS[hssV1+(offset * num_of_constants)] = 82.9;
CONSTANTS[hssV2+(offset * num_of_constants)] = 6.086;
CONSTANTS[Ahf+(offset * num_of_constants)] = 0.99;
CONSTANTS[Ahs+(offset * num_of_constants)] = 1.00000 - CONSTANTS[Ahf+(offset * num_of_constants)];
CONSTANTS[GNaL_b+(offset * num_of_constants)] = 0.0075;
CONSTANTS[GNaL+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[GNaL_b+(offset * num_of_constants)]*0.600000 : CONSTANTS[GNaL_b+(offset * num_of_constants)]);
CONSTANTS[thL+(offset * num_of_constants)] = 200;
CONSTANTS[thLp+(offset * num_of_constants)] =  3.00000*CONSTANTS[thL+(offset * num_of_constants)];
CONSTANTS[PNab+(offset * num_of_constants)] = 3.75e-10;
CONSTANTS[Gto_b+(offset * num_of_constants)] = 0.02;
CONSTANTS[Gto+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[Gto_b+(offset * num_of_constants)]*4.00000 : CONSTANTS[celltype+(offset * num_of_constants)]==2.00000 ?  CONSTANTS[Gto_b+(offset * num_of_constants)]*4.00000 : CONSTANTS[Gto_b+(offset * num_of_constants)]);
CONSTANTS[ko+(offset * num_of_constants)] = 5.4;
CONSTANTS[GKr_b+(offset * num_of_constants)] = 0.046;
CONSTANTS[GKr+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[GKr_b+(offset * num_of_constants)]*1.30000 : CONSTANTS[celltype+(offset * num_of_constants)]==2.00000 ?  CONSTANTS[GKr_b+(offset * num_of_constants)]*0.800000 : CONSTANTS[GKr_b+(offset * num_of_constants)]);
CONSTANTS[GKs_b+(offset * num_of_constants)] = 0.0034;
CONSTANTS[GKs+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[GKs_b+(offset * num_of_constants)]*1.40000 : CONSTANTS[GKs_b+(offset * num_of_constants)]);
CONSTANTS[PKNa+(offset * num_of_constants)] = 0.01833;
CONSTANTS[GK1_b+(offset * num_of_constants)] = 0.1908;
CONSTANTS[GK1+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[GK1_b+(offset * num_of_constants)]*1.20000 : CONSTANTS[celltype+(offset * num_of_constants)]==2.00000 ?  CONSTANTS[GK1_b+(offset * num_of_constants)]*1.30000 : CONSTANTS[GK1_b+(offset * num_of_constants)]);
CONSTANTS[GKb_b+(offset * num_of_constants)] = 0.003;
CONSTANTS[GKb+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[GKb_b+(offset * num_of_constants)]*0.600000 : CONSTANTS[GKb_b+(offset * num_of_constants)]);
CONSTANTS[Kmn+(offset * num_of_constants)] = 0.002;
CONSTANTS[k2n+(offset * num_of_constants)] = 1000;
CONSTANTS[tjca+(offset * num_of_constants)] = 75.0000;
CONSTANTS[Aff+(offset * num_of_constants)] = 0.600000;
CONSTANTS[Afs+(offset * num_of_constants)] = 1.00000 - CONSTANTS[Aff+(offset * num_of_constants)];
CONSTANTS[PCa_b+(offset * num_of_constants)] = 0.0001;
CONSTANTS[PCa+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[PCa_b+(offset * num_of_constants)]*1.20000 : CONSTANTS[celltype+(offset * num_of_constants)]==2.00000 ?  CONSTANTS[PCa_b+(offset * num_of_constants)]*2.50000 : CONSTANTS[PCa_b+(offset * num_of_constants)]);
CONSTANTS[PCaK+(offset * num_of_constants)] =  0.000357400*CONSTANTS[PCa+(offset * num_of_constants)];
CONSTANTS[PCaNa+(offset * num_of_constants)] =  0.00125000*CONSTANTS[PCa+(offset * num_of_constants)];
CONSTANTS[PCap+(offset * num_of_constants)] =  1.10000*CONSTANTS[PCa+(offset * num_of_constants)];
CONSTANTS[PCaKp+(offset * num_of_constants)] =  0.000357400*CONSTANTS[PCap+(offset * num_of_constants)];
CONSTANTS[PCaNap+(offset * num_of_constants)] =  0.00125000*CONSTANTS[PCap+(offset * num_of_constants)];
CONSTANTS[cao+(offset * num_of_constants)] = 1.8;
CONSTANTS[PCab+(offset * num_of_constants)] = 2.5e-8;
CONSTANTS[GpCa+(offset * num_of_constants)] = 0.0005;
CONSTANTS[KmCap+(offset * num_of_constants)] = 0.0005;
CONSTANTS[kasymm+(offset * num_of_constants)] = 12.5;
CONSTANTS[kcaon+(offset * num_of_constants)] = 1.5e6;
CONSTANTS[kcaoff+(offset * num_of_constants)] = 5e3;
CONSTANTS[kna1+(offset * num_of_constants)] = 15;
CONSTANTS[kna2+(offset * num_of_constants)] = 5;
CONSTANTS[kna3+(offset * num_of_constants)] = 88.12;
CONSTANTS[qna+(offset * num_of_constants)] = 0.5224;
CONSTANTS[qca+(offset * num_of_constants)] = 0.167;
CONSTANTS[wnaca+(offset * num_of_constants)] = 5e3;
CONSTANTS[wna+(offset * num_of_constants)] = 6e4;
CONSTANTS[wca+(offset * num_of_constants)] = 6e4;
CONSTANTS[KmCaAct+(offset * num_of_constants)] = 150e-6;
CONSTANTS[Gncx_b+(offset * num_of_constants)] = 0.0008;
CONSTANTS[Gncx+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[Gncx_b+(offset * num_of_constants)]*1.10000 : CONSTANTS[celltype+(offset * num_of_constants)]==2.00000 ?  CONSTANTS[Gncx_b+(offset * num_of_constants)]*1.40000 : CONSTANTS[Gncx_b+(offset * num_of_constants)]);
CONSTANTS[h10_i+(offset * num_of_constants)] = CONSTANTS[kasymm+(offset * num_of_constants)]+1.00000+ (CONSTANTS[nao+(offset * num_of_constants)]/CONSTANTS[kna1+(offset * num_of_constants)])*(1.00000+CONSTANTS[nao+(offset * num_of_constants)]/CONSTANTS[kna2+(offset * num_of_constants)]);
CONSTANTS[h11_i+(offset * num_of_constants)] = ( CONSTANTS[nao+(offset * num_of_constants)]*CONSTANTS[nao+(offset * num_of_constants)])/( CONSTANTS[h10_i+(offset * num_of_constants)]*CONSTANTS[kna1+(offset * num_of_constants)]*CONSTANTS[kna2+(offset * num_of_constants)]);
CONSTANTS[h12_i+(offset * num_of_constants)] = 1.00000/CONSTANTS[h10_i+(offset * num_of_constants)];
CONSTANTS[k1_i+(offset * num_of_constants)] =  CONSTANTS[h12_i+(offset * num_of_constants)]*CONSTANTS[cao+(offset * num_of_constants)]*CONSTANTS[kcaon+(offset * num_of_constants)];
CONSTANTS[k2_i+(offset * num_of_constants)] = CONSTANTS[kcaoff+(offset * num_of_constants)];
CONSTANTS[k5_i+(offset * num_of_constants)] = CONSTANTS[kcaoff+(offset * num_of_constants)];
CONSTANTS[h10_ss+(offset * num_of_constants)] = CONSTANTS[kasymm+(offset * num_of_constants)]+1.00000+ (CONSTANTS[nao+(offset * num_of_constants)]/CONSTANTS[kna1+(offset * num_of_constants)])*(1.00000+CONSTANTS[nao+(offset * num_of_constants)]/CONSTANTS[kna2+(offset * num_of_constants)]);
CONSTANTS[h11_ss+(offset * num_of_constants)] = ( CONSTANTS[nao+(offset * num_of_constants)]*CONSTANTS[nao+(offset * num_of_constants)])/( CONSTANTS[h10_ss+(offset * num_of_constants)]*CONSTANTS[kna1+(offset * num_of_constants)]*CONSTANTS[kna2+(offset * num_of_constants)]);
CONSTANTS[h12_ss+(offset * num_of_constants)] = 1.00000/CONSTANTS[h10_ss+(offset * num_of_constants)];
CONSTANTS[k1_ss+(offset * num_of_constants)] =  CONSTANTS[h12_ss+(offset * num_of_constants)]*CONSTANTS[cao+(offset * num_of_constants)]*CONSTANTS[kcaon+(offset * num_of_constants)];
CONSTANTS[k2_ss+(offset * num_of_constants)] = CONSTANTS[kcaoff+(offset * num_of_constants)];
CONSTANTS[k5_ss+(offset * num_of_constants)] = CONSTANTS[kcaoff+(offset * num_of_constants)];
CONSTANTS[k1p+(offset * num_of_constants)] = 949.5;
CONSTANTS[k2p+(offset * num_of_constants)] = 687.2;
CONSTANTS[k3p+(offset * num_of_constants)] = 1899;
CONSTANTS[k4p+(offset * num_of_constants)] = 639;
CONSTANTS[k1m+(offset * num_of_constants)] = 182.4;
CONSTANTS[k2m+(offset * num_of_constants)] = 39.4;
CONSTANTS[k3m+(offset * num_of_constants)] = 79300;
CONSTANTS[k4m+(offset * num_of_constants)] = 40;
CONSTANTS[Knai0+(offset * num_of_constants)] = 9.073;
CONSTANTS[Knao0+(offset * num_of_constants)] = 27.78;
CONSTANTS[delta+(offset * num_of_constants)] = -0.155;
CONSTANTS[Kki+(offset * num_of_constants)] = 0.5;
CONSTANTS[Kko+(offset * num_of_constants)] = 0.3582;
CONSTANTS[MgADP+(offset * num_of_constants)] = 0.05;
CONSTANTS[MgATP+(offset * num_of_constants)] = 9.8;
CONSTANTS[H+(offset * num_of_constants)] = 1e-7;
CONSTANTS[Kmgatp+(offset * num_of_constants)] = 1.698e-7;
CONSTANTS[eP+(offset * num_of_constants)] = 4.2;
CONSTANTS[Khp+(offset * num_of_constants)] = 1.698e-7;
CONSTANTS[Knap+(offset * num_of_constants)] = 224;
CONSTANTS[Kxkur+(offset * num_of_constants)] = 292;
CONSTANTS[a2+(offset * num_of_constants)] = CONSTANTS[k2p+(offset * num_of_constants)];
CONSTANTS[a4+(offset * num_of_constants)] = (( CONSTANTS[k4p+(offset * num_of_constants)]*CONSTANTS[MgATP+(offset * num_of_constants)])/CONSTANTS[Kmgatp+(offset * num_of_constants)])/(1.00000+CONSTANTS[MgATP+(offset * num_of_constants)]/CONSTANTS[Kmgatp+(offset * num_of_constants)]);
CONSTANTS[b1+(offset * num_of_constants)] =  CONSTANTS[k1m+(offset * num_of_constants)]*CONSTANTS[MgADP+(offset * num_of_constants)];
CONSTANTS[Pnak_b+(offset * num_of_constants)] = 30;
CONSTANTS[Pnak+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[Pnak_b+(offset * num_of_constants)]*0.900000 : CONSTANTS[celltype+(offset * num_of_constants)]==2.00000 ?  CONSTANTS[Pnak_b+(offset * num_of_constants)]*0.700000 : CONSTANTS[Pnak_b+(offset * num_of_constants)]);
CONSTANTS[upScale+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[bt+(offset * num_of_constants)] = 4.75;
CONSTANTS[btp+(offset * num_of_constants)] =  1.25000*CONSTANTS[bt+(offset * num_of_constants)];
CONSTANTS[a_relp+(offset * num_of_constants)] =  0.500000*CONSTANTS[btp+(offset * num_of_constants)];
CONSTANTS[a_rel+(offset * num_of_constants)] =  0.500000*CONSTANTS[bt+(offset * num_of_constants)];
CONSTANTS[aCaMK+(offset * num_of_constants)] = 0.05;
CONSTANTS[bCaMK+(offset * num_of_constants)] = 0.00068;
CONSTANTS[Ageo+(offset * num_of_constants)] =  2.00000*3.14000*CONSTANTS[rad+(offset * num_of_constants)]*CONSTANTS[rad+(offset * num_of_constants)]+ 2.00000*3.14000*CONSTANTS[rad+(offset * num_of_constants)]*CONSTANTS[L+(offset * num_of_constants)];
CONSTANTS[Acap+(offset * num_of_constants)] =  2.00000*CONSTANTS[Ageo+(offset * num_of_constants)];
CONSTANTS[vmyo+(offset * num_of_constants)] =  0.680000*CONSTANTS[vcell+(offset * num_of_constants)];
CONSTANTS[vss+(offset * num_of_constants)] =  0.0200000*CONSTANTS[vcell+(offset * num_of_constants)];
CONSTANTS[vjsr+(offset * num_of_constants)] =  0.00480000*CONSTANTS[vcell+(offset * num_of_constants)];
CONSTANTS[vnsr+(offset * num_of_constants)] =  0.0552000*CONSTANTS[vcell+(offset * num_of_constants)];
CONSTANTS[cmdnmax_b+(offset * num_of_constants)] = 0.05;
CONSTANTS[cmdnmax+(offset * num_of_constants)] = (CONSTANTS[celltype+(offset * num_of_constants)]==1.00000 ?  CONSTANTS[cmdnmax_b+(offset * num_of_constants)]*1.30000 : CONSTANTS[cmdnmax_b+(offset * num_of_constants)]);
CONSTANTS[kmcmdn+(offset * num_of_constants)] = 0.00238;
CONSTANTS[trpnmax+(offset * num_of_constants)] = 0.07;
CONSTANTS[kmtrpn+(offset * num_of_constants)] = 0.0005;
CONSTANTS[BSLmax+(offset * num_of_constants)] = 1.124;
CONSTANTS[BSRmax+(offset * num_of_constants)] = 0.047;
CONSTANTS[KmBSR+(offset * num_of_constants)] = 0.00087;
CONSTANTS[KmBSL+(offset * num_of_constants)] = 0.0087;
CONSTANTS[csqnmax+(offset * num_of_constants)] = 10;
CONSTANTS[kmcsqn+(offset * num_of_constants)] = 0.8;
STATES[m+(offset * num_of_states)] = 0;
STATES[j+(offset * num_of_states)] = 1;
STATES[jp+(offset * num_of_states)] = 1;
STATES[hf+(offset * num_of_states)] = 1;
STATES[hs+(offset * num_of_states)] = 1;
STATES[hsp+(offset * num_of_states)] = 1;
STATES[V+(offset * num_of_states)] = -87;
STATES[CaMKt+(offset * num_of_states)] = 0;
STATES[cass+(offset * num_of_states)] = 1e-4;

STATES[nai+(offset * num_of_states)] = 7;
STATES[mL+(offset * num_of_states)] = 0;

STATES[hL+(offset * num_of_states)] = 1;
STATES[hLp+(offset * num_of_states)] = 1;
STATES[a+(offset * num_of_states)] = 0;
STATES[ap+(offset * num_of_states)] = 0;
STATES[ki+(offset * num_of_states)] = 145;
STATES[iF+(offset * num_of_states)] = 1;
STATES[iS+(offset * num_of_states)] = 1;
STATES[iFp+(offset * num_of_states)] = 1;
STATES[iSp+(offset * num_of_states)] = 1;
STATES[xrf+(offset * num_of_states)] = 0;
STATES[xrs+(offset * num_of_states)] = 0;
STATES[xs1+(offset * num_of_states)] = 0;
STATES[xs2+(offset * num_of_states)] = 0;
STATES[cai+(offset * num_of_states)] = 1e-4;
STATES[xk1+(offset * num_of_states)] = 1;
STATES[d+(offset * num_of_states)] = 0;
STATES[ff+(offset * num_of_states)] = 1;
STATES[fs+(offset * num_of_states)] = 1;
STATES[fcaf+(offset * num_of_states)] = 1;
STATES[nca+(offset * num_of_states)] = 0;
STATES[jca+(offset * num_of_states)] = 1;
STATES[fcas+(offset * num_of_states)] = 1;
STATES[ffp+(offset * num_of_states)] = 1;
STATES[fcafp+(offset * num_of_states)] = 1;
STATES[kss+(offset * num_of_states)] = 145;
STATES[nass+(offset * num_of_states)] = 7;
STATES[cansr+(offset * num_of_states)] = 1.2;
STATES[Jrelnp+(offset * num_of_states)] = 0;
STATES[Jrelp+(offset * num_of_states)] = 0;
STATES[cajsr+(offset * num_of_states)] = 1.2;
// for (int zz = 0; zz < num_of_constants; zz++){
//   printf("core %d, constants[%d] = %lf\n", offset, zz, CONSTANTS[zz]);
// }
// for (int aa = 0; aa < num_of_states; aa++){
//   printf("core %d, states[%d] = %lf\n", offset, aa, STATES[aa]);
// }
}

__device__ double set_time_step(
    double TIME,
    double time_point,
    double max_time_step,
    double* CONSTANTS,
    double* RATES) 

    {
    double time_step = 0.005;
    // int offset = threadIdx.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int num_of_constants = 146;
    int num_of_rates = 41; 
    // core selain 0 banyak masuk ke if ini, jadi time step nya stuck 
    // di semua core pun, kalo semisal tcurr (TIME) udah di atas time_point (25.0), harusnya logika kedua bernilai false biar ada update time_step
    // core selain 0 sepertinya selalu true buat logika kedua.
    // koreksi, gak ternyata, ternyata core2 lain gak masuk ke if ini, udah masuk else, 
    if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[stim_period + (offset * num_of_constants)]) * CONSTANTS[stim_period + (offset * num_of_constants)] ) <= time_point) {
        // printf("TIME <= time_point ms for core %d \n", offset);
        //printf(" untuk core %d masuk if pertama dengan nilai perhitungan sisi kedua: %lf\n", offset, (TIME - floor(TIME / CONSTANTS[stim_period + (offset * num_of_constants)]) * CONSTANTS[stim_period + (offset * num_of_constants)] ));
          // printf("dV = %lf, time_step = %lf\n",RATES[V] * time_step, time_step);
        return time_step;
      
    }
    else {
          //printf("core %d gak masuk if pertama, jadi masuk else, dengan nilai kondisional %lf \n", offset, std::abs(RATES[V + (offset * num_of_rates)] * time_step));
          //printf("nilai rates di core %d: %lf\n",offset, RATES[V + (offset * num_of_rates)]);
        //printf("TIME > time_point ms\n");
        if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) <= 0.2) {//Slow changes in V
        // hanya core 0 yang berhasil masuk ke if ini
            // printf("dV/dt <= 0.2 for core %d\n", offset);
            time_step = std::abs(0.8 / RATES[V + (offset * num_of_rates)] );
            //printf("untuk core %d masuk if kedua dengan nilai time step %lf\n", offset, time_step);
            //Make sure time_step is between 0.005 and max_time_step
            if (time_step < 0.005) {
                time_step = 0.005;
            }
            else if (time_step > max_time_step) {
                time_step = max_time_step;
            }
            // printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
        }
        else if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) >= 0.8) {//Fast changes in V
        // core lain gak pernah masuk sini! Bahkan gak pernah ada yang masuk ke sini, 
            //printf("dV/dt >= 0.8\n");
            time_step = std::abs(0.2 / RATES[V+ (offset * num_of_rates)]);
            //printf("untuk core %d masuk else dari if kedua dengan nilai time step %lf\n", offset, time_step);
            while (std::abs(RATES[V+ (offset * num_of_rates)] * time_step) >= 0.8 && 0.005 < time_step && time_step < max_time_step) {
                time_step = time_step / 10.0;
                // printf("dV = %lf, time_step = %lf\n",std::abs(RATES[V] * time_step), time_step);
            }
        }
        __syncthreads();
        return time_step;
    }
}



__device__ void computeRates(double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC)
{
int num_of_constants = 146;
int num_of_states = 41;
int num_of_algebraic = 199;
int num_of_rates = 41;
// int offset = threadIdx.x; 
int offset = blockIdx.x * blockDim.x + threadIdx.x;

ALGEBRAIC[vffrt + (offset * num_of_algebraic)] = ( STATES[V + (offset * num_of_states)]*CONSTANTS[F+ (offset * num_of_constants)]*CONSTANTS[F+ (offset * num_of_constants)])/( CONSTANTS[R+ (offset * num_of_constants)]*CONSTANTS[T+ (offset * num_of_constants)]);
ALGEBRAIC[vfrt + (offset * num_of_algebraic)] = ( STATES[V+ (offset * num_of_states)]*CONSTANTS[F+ (offset * num_of_constants)])/( CONSTANTS[R + (offset * num_of_constants)]*CONSTANTS[T + (offset * num_of_constants)]);
ALGEBRAIC[Istim+ (offset * num_of_algebraic)] = (TIME>=CONSTANTS[stim_start + (offset * num_of_constants)]&&TIME<=CONSTANTS[stim_end+ (offset * num_of_constants)]&&(TIME - CONSTANTS[stim_start + (offset * num_of_constants)]) -  floor((TIME - CONSTANTS[stim_start + (offset * num_of_constants)])/CONSTANTS[stim_period+ (offset * num_of_constants)])*CONSTANTS[stim_period + (offset * num_of_constants)]<=CONSTANTS[duration + (offset * num_of_constants)] ? CONSTANTS[amp + (offset * num_of_constants)] : 0.00000);
ALGEBRAIC[mss + (offset * num_of_algebraic)] = 1.00000/ (1.00000+exp(- (STATES[V + (offset * num_of_states)]+CONSTANTS[mssV1+ (offset * num_of_constants)])/CONSTANTS[mssV2 + (offset * num_of_constants)]));
ALGEBRAIC[tm + (offset * num_of_algebraic)] = 1.00000/( CONSTANTS[mtD1 + (offset * num_of_constants)]*exp((STATES[V + (offset * num_of_states)]+CONSTANTS[mtV1 + (offset * num_of_constants)])/CONSTANTS[mtV2 + (offset * num_of_constants)])+ CONSTANTS[mtD2 + (offset * num_of_constants) ]*exp(- (STATES[V + (offset * num_of_states) ]+CONSTANTS[mtV3 + (offset * num_of_constants)])/CONSTANTS[mtV4 + (offset * num_of_constants)]));
ALGEBRAIC[hss + (offset * num_of_algebraic)] = 1.00000/(1.00000+exp((STATES[V + (offset * num_of_states)]+CONSTANTS[hssV1 + (offset * num_of_constants)])/CONSTANTS[hssV2 + (offset * num_of_constants)]));
ALGEBRAIC[ths + (offset * num_of_algebraic) ] = 1.00000/( 0.00979400*exp(- (STATES[V + (offset * num_of_states) ]+17.9500)/28.0500)+ 0.334300*exp((STATES[V + (offset * num_of_states) ]+5.73000)/56.6600));
ALGEBRAIC[thf + (offset * num_of_algebraic) ] = 1.00000/( 1.43200e-05*exp(- (STATES[V + (offset * num_of_states) ]+1.19600)/6.28500)+ 6.14900*exp((STATES[V + (offset * num_of_states) ]+0.509600)/20.2700));
ALGEBRAIC[h + (offset * num_of_algebraic) ] =  CONSTANTS[Ahf + (offset * num_of_constants) ]*STATES[hf+ (offset * num_of_states) ]+ CONSTANTS[Ahs + (offset * num_of_constants)]*STATES[hs + (offset * num_of_states)];
ALGEBRAIC[jss + (offset * num_of_algebraic) ] = ALGEBRAIC[hss + (offset * num_of_algebraic)];
ALGEBRAIC[tj + (offset * num_of_algebraic) ] = 2.03800+1.00000/( 0.0213600*exp(- (STATES[V + (offset * num_of_states)]+100.600)/8.28100)+ 0.305200*exp((STATES[V + (offset * num_of_states)]+0.994100)/38.4500));
ALGEBRAIC[hssp + (offset * num_of_algebraic) ] = 1.00000/(1.00000+exp((STATES[V+ (offset * num_of_states)]+89.1000)/6.08600));
ALGEBRAIC[thsp + (offset * num_of_algebraic)] =  3.00000*ALGEBRAIC[ths+ (offset * num_of_algebraic)];
ALGEBRAIC[hp + (offset * num_of_algebraic)] =  CONSTANTS[Ahf+ (offset * num_of_constants)]*STATES[hf+ (offset * num_of_states)]+ CONSTANTS[Ahs + (offset * num_of_constants)]*STATES[hsp + (offset * num_of_states)];
ALGEBRAIC[tjp + (offset * num_of_algebraic)] =  1.46000*ALGEBRAIC[tj + (offset * num_of_algebraic)];

ALGEBRAIC[ENa + (offset * num_of_algebraic) ] =  (( CONSTANTS[R + (offset * num_of_constants) ]*CONSTANTS[T + (offset * num_of_constants) ])/CONSTANTS[F + (offset * num_of_constants) ])*log(CONSTANTS[nao + (offset * num_of_constants)]/STATES[nai + (offset * num_of_states)]);
// printf("core %d, uraian ena: (( %lf * %lf ) / %lf ) * log( %lf / %lf );\n",offset, CONSTANTS[R + (offset * num_of_constants)],CONSTANTS[T + (offset * num_of_constants)], CONSTANTS[F + (offset * num_of_constants)], CONSTANTS[nao + (offset * num_of_constants)], STATES[nai + (offset * num_of_states)]);

ALGEBRAIC[CaMKb + (offset * num_of_algebraic) ] = ( CONSTANTS[CaMKo + (offset * num_of_constants) ]*(1.00000 - STATES[CaMKt + (offset * num_of_states)]))/(1.00000+CONSTANTS[KmCaM + (offset * num_of_constants)]/STATES[cass + (offset * num_of_states)]);
ALGEBRAIC[CaMKa + (offset * num_of_algebraic) ] = ALGEBRAIC[CaMKb + (offset * num_of_algebraic) ]+STATES[CaMKt + (offset * num_of_states)];
ALGEBRAIC[fINap + (offset * num_of_algebraic) ] = 1.00000/(1.00000+CONSTANTS[KmCaMK + (offset * num_of_constants)]/ALGEBRAIC[CaMKa + (offset * num_of_algebraic)]);

ALGEBRAIC[INa + (offset * num_of_algebraic) ] =  CONSTANTS[GNa + (offset * num_of_constants)]*(STATES[V + (offset * num_of_states)] - ALGEBRAIC[ENa + (offset * num_of_algebraic)])*pow(STATES[m + (offset * num_of_states)], 3.00000)*( (1.00000 - ALGEBRAIC[fINap + (offset * num_of_algebraic)])*ALGEBRAIC[h + (offset * num_of_algebraic)]*STATES[j + (offset * num_of_states)]+ ALGEBRAIC[fINap + (offset * num_of_algebraic)]*ALGEBRAIC[hp+ (offset * num_of_algebraic)]*STATES[jp + (offset * num_of_states)]);
//printf("core %d, uraian algebraic ina: %lf * (%lf - %lf) * pow(%lf, 3.00000)*( (1.00000 - %lf) * %lf * %lf + %lf * %lf * %lf)\n", offset, CONSTANTS[GNa + (offset * num_of_constants)], STATES[V + (offset * num_of_states)], ALGEBRAIC[ENa + (offset * num_of_algebraic)],STATES[m + (offset * num_of_states)], ALGEBRAIC[fINap + (offset * num_of_algebraic)], ALGEBRAIC[h + (offset * num_of_algebraic)], STATES[j + (offset * num_of_states)],  ALGEBRAIC[fINap + (offset * num_of_algebraic)], ALGEBRAIC[hp+ (offset * num_of_algebraic)], STATES[jp + (offset * num_of_states)]);

ALGEBRAIC[mLss + (offset * num_of_algebraic) ] = 1.00000/(1.00000+exp(- (STATES[V + (offset * num_of_states)]+42.8500)/5.26400));
ALGEBRAIC[tmL + (offset * num_of_algebraic)] = ALGEBRAIC[tm + (offset * num_of_algebraic)];
ALGEBRAIC[hLss + (offset * num_of_algebraic)] = 1.00000/(1.00000+exp((STATES[V + (offset * num_of_states) ]+87.6100)/7.48800));
ALGEBRAIC[hLssp + (offset * num_of_algebraic) ] = 1.00000/(1.00000+exp((STATES[V + (offset * num_of_states)]+93.8100)/7.48800));
ALGEBRAIC[fINaLp + (offset * num_of_algebraic)] = 1.00000/(1.00000+CONSTANTS[KmCaMK + (offset * num_of_constants)]/ALGEBRAIC[CaMKa + (offset * num_of_algebraic)]);

ALGEBRAIC[INaL + (offset * num_of_algebraic) ] =  CONSTANTS[GNaL + (offset * num_of_constants) ]*(STATES[V + (offset * num_of_states)] - ALGEBRAIC[ENa + (offset * num_of_algebraic)])*STATES[mL + (offset * num_of_states) ]*( (1.00000 - ALGEBRAIC[fINaLp + (offset * num_of_algebraic)])*STATES[hL + (offset * num_of_states) ]+ ALGEBRAIC[fINaLp + (offset * num_of_algebraic) ]*STATES[hLp + (offset * num_of_states) ]);

ALGEBRAIC[INab + (offset * num_of_algebraic) ] = ( CONSTANTS[PNab + (offset * num_of_constants)]*ALGEBRAIC[vffrt + (offset * num_of_algebraic) ]*(STATES[nai + (offset * num_of_states)]*exp(ALGEBRAIC[vfrt + (offset * num_of_algebraic)]) - CONSTANTS[nao + (offset * num_of_constants)]))/(exp(ALGEBRAIC[vfrt + (offset * num_of_algebraic)]) - 1.00000);
// nge ubah states nya di sini tadi, yang gak ke offset
ALGEBRAIC[ass + (offset * num_of_algebraic) ] = 1.00000/(1.00000+exp(- (STATES[V + (offset * num_of_states)] - 14.3400)/14.8200));
ALGEBRAIC[ta + (offset * num_of_algebraic) ] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- (STATES[V + (offset * num_of_states)] - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[V + (offset * num_of_states)]+100.000)/29.3814)));
ALGEBRAIC[iss + (offset * num_of_algebraic) ] = 1.00000/(1.00000+exp((STATES[V+ (offset * num_of_states)]+43.9400)/5.71100));
ALGEBRAIC[delta_epi + (offset * num_of_algebraic) ] = (CONSTANTS[celltype + (offset * num_of_constants)]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[V + (offset * num_of_states)]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[tiF_b + (offset * num_of_algebraic) ] = 4.56200+1.00000/( 0.393300*exp(- (STATES[V + (offset * num_of_states) ]+100.000)/100.000)+ 0.0800400*exp((STATES[V + (offset * num_of_states) ]+50.0000)/16.5900));
ALGEBRAIC[tiS_b + (offset * num_of_algebraic) ] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[V + (offset * num_of_states)]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[V + (offset * num_of_states)]+114.100)/8.07900));
ALGEBRAIC[tiF + (offset * num_of_algebraic)] =  ALGEBRAIC[tiF_b+ (offset * num_of_algebraic)]*ALGEBRAIC[delta_epi+ (offset * num_of_algebraic)];
ALGEBRAIC[tiS + (offset * num_of_algebraic)] =  ALGEBRAIC[tiS_b + (offset * num_of_algebraic) ]*ALGEBRAIC[delta_epi + (offset * num_of_algebraic)];
ALGEBRAIC[AiF + (offset * num_of_algebraic)] = 1.00000/(1.00000+exp((STATES[V + (offset * num_of_states)] - 213.600)/151.200));
ALGEBRAIC[AiS + (offset * num_of_algebraic)] = 1.00000 - ALGEBRAIC[AiF + (offset * num_of_algebraic)];
ALGEBRAIC[i + (offset * num_of_algebraic)] =  ALGEBRAIC[AiF + (offset * num_of_algebraic) ]*STATES[iF + (offset * num_of_states) ]+ ALGEBRAIC[AiS + (offset * num_of_algebraic)]*STATES[iS + (offset * num_of_states)];
ALGEBRAIC[assp + (offset * num_of_algebraic)] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V] - 24.3400)/14.8200));

ALGEBRAIC[(offset * num_of_algebraic) + dti_develop] = 1.35400+0.000100000/(exp((STATES[(offset * num_of_states) + V] - 167.400)/15.8900)+exp(- (STATES[(offset * num_of_states) + V] - 12.2300)/0.215400));
ALGEBRAIC[(offset * num_of_algebraic) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(offset * num_of_states) + V]+70.0000)/20.0000));
ALGEBRAIC[(offset * num_of_algebraic) + tiFp] = ALGEBRAIC[(offset * num_of_algebraic) + dti_develop] * ALGEBRAIC[(offset * num_of_algebraic) + dti_recover] * ALGEBRAIC[(offset * num_of_algebraic) + tiF];
ALGEBRAIC[(offset * num_of_algebraic) + tiSp] = ALGEBRAIC[(offset * num_of_algebraic) + dti_develop] * ALGEBRAIC[(offset * num_of_algebraic) + dti_recover] * ALGEBRAIC[(offset * num_of_algebraic) + tiS];
ALGEBRAIC[(offset * num_of_algebraic) + ip] =  ALGEBRAIC[(offset * num_of_algebraic) + AiF]*STATES[(offset * num_of_states) + iFp]+ ALGEBRAIC[(offset * num_of_algebraic) + AiS]*STATES[(offset * num_of_states) + iSp];
ALGEBRAIC[(offset * num_of_algebraic) + EK] =  (( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T])/CONSTANTS[(offset * num_of_constants) + F])*log(CONSTANTS[(offset * num_of_constants) + ko]/STATES[(offset * num_of_states) + ki]);
ALGEBRAIC[(offset * num_of_algebraic) + fItop] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + Ito] =  CONSTANTS[(offset * num_of_constants) + Gto]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK])*( (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fItop])*STATES[(offset * num_of_states) + a]*ALGEBRAIC[(offset * num_of_algebraic) + i]+ ALGEBRAIC[(offset * num_of_algebraic) + fItop]*STATES[(offset * num_of_states) + ap]*ALGEBRAIC[(offset * num_of_algebraic) + ip]);
ALGEBRAIC[(offset * num_of_algebraic) + xrss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+8.33700)/6.78900));
ALGEBRAIC[(offset * num_of_algebraic) + txrf] = 12.9800+1.00000/( 0.365200*exp((STATES[(offset * num_of_states) + V] - 31.6600)/3.86900)+ 4.12300e-05*exp(- (STATES[(offset * num_of_states) + V] - 47.7800)/20.3800));
ALGEBRAIC[(offset * num_of_algebraic) + txrs] = 1.86500+1.00000/( 0.0662900*exp((STATES[(offset * num_of_states) + V] - 34.7000)/7.35500)+ 1.12800e-05*exp(- (STATES[(offset * num_of_states) + V] - 29.7400)/25.9400));
ALGEBRAIC[(offset * num_of_algebraic) + Axrf] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+54.8100)/38.2100));
ALGEBRAIC[(offset * num_of_algebraic) + Axrs] = 1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + Axrf];
ALGEBRAIC[(offset * num_of_algebraic) + xr] =  ALGEBRAIC[(offset * num_of_algebraic) + Axrf]*STATES[(offset * num_of_states) + xrf]+ ALGEBRAIC[(offset * num_of_algebraic) + Axrs]*STATES[(offset * num_of_states) + xrs];
ALGEBRAIC[(offset * num_of_algebraic) + rkr] = ( (1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+55.0000)/75.0000)))*1.00000)/(1.00000+exp((STATES[(offset * num_of_states) + V] - 10.0000)/30.0000));
ALGEBRAIC[(offset * num_of_algebraic) + IKr] =  CONSTANTS[(offset * num_of_constants) + GKr]* pow((CONSTANTS[(offset * num_of_constants) + ko]/5.40000), 1.0 / 2)*ALGEBRAIC[(offset * num_of_algebraic) + xr]*ALGEBRAIC[(offset * num_of_algebraic) + rkr]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK]);
ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+11.6000)/8.93200));
ALGEBRAIC[(offset * num_of_algebraic) + txs1] = 817.300+1.00000/( 0.000232600*exp((STATES[(offset * num_of_states) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(offset * num_of_states) + V]+210.000)/230.000));
ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] = ALGEBRAIC[(offset * num_of_algebraic) + xs1ss];
ALGEBRAIC[(offset * num_of_algebraic) + txs2] = 1.00000/( 0.0100000*exp((STATES[(offset * num_of_states) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(offset * num_of_states) + V]+66.5400)/31.0000));
ALGEBRAIC[(offset * num_of_algebraic) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(offset * num_of_states) + cai], 1.40000));
ALGEBRAIC[(offset * num_of_algebraic) + EKs] =  (( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T])/CONSTANTS[(offset * num_of_constants) + F])*log((CONSTANTS[(offset * num_of_constants) + ko]+ CONSTANTS[(offset * num_of_constants) + PKNa]*CONSTANTS[(offset * num_of_constants) + nao])/(STATES[(offset * num_of_states) + ki]+ CONSTANTS[(offset * num_of_constants) + PKNa]*STATES[(offset * num_of_states) + nai]));
ALGEBRAIC[(offset * num_of_algebraic) + IKs] =  CONSTANTS[(offset * num_of_constants) + GKs]*ALGEBRAIC[(offset * num_of_algebraic) + KsCa]*STATES[(offset * num_of_states) + xs1]*STATES[(offset * num_of_states) + xs2]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EKs]);
ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+ 2.55380*CONSTANTS[(offset * num_of_constants) + ko]+144.590)/( 1.56920*CONSTANTS[(offset * num_of_constants) + ko]+3.81150)));
ALGEBRAIC[(offset * num_of_algebraic) + txk1] = 122.200/(exp(- (STATES[(offset * num_of_states) + V]+127.200)/20.3600)+exp((STATES[(offset * num_of_states) + V]+236.800)/69.3300));
ALGEBRAIC[(offset * num_of_algebraic) + rk1] = 1.00000/(1.00000+exp(((STATES[(offset * num_of_states) + V]+105.800) -  2.60000*CONSTANTS[(offset * num_of_constants) + ko])/9.49300));
ALGEBRAIC[(offset * num_of_algebraic) + IK1] =  CONSTANTS[(offset * num_of_constants) + GK1]* pow(CONSTANTS[(offset * num_of_constants) + ko], 1.0 / 2)*ALGEBRAIC[(offset * num_of_algebraic) + rk1]*STATES[(offset * num_of_states) + xk1]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK]);
ALGEBRAIC[(offset * num_of_algebraic) + xkb] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V] - 14.4800)/18.3400));
ALGEBRAIC[(offset * num_of_algebraic) + IKb] =  CONSTANTS[(offset * num_of_constants) + GKb]*ALGEBRAIC[(offset * num_of_algebraic) + xkb]*(STATES[(offset * num_of_states) + V] - ALGEBRAIC[(offset * num_of_algebraic) + EK]);
ALGEBRAIC[(offset * num_of_algebraic) + dss] = 1.00000/(1.00000+exp(- (STATES[(offset * num_of_states) + V]+3.94000)/4.23000));
ALGEBRAIC[(offset * num_of_algebraic) + td] = 0.600000+1.00000/(exp( - 0.0500000*(STATES[(offset * num_of_states) + V]+6.00000))+exp( 0.0900000*(STATES[(offset * num_of_states) + V]+14.0000)));
ALGEBRAIC[(offset * num_of_algebraic) + fss] = 1.00000/(1.00000+exp((STATES[(offset * num_of_states) + V]+19.5800)/3.69600));
ALGEBRAIC[(offset * num_of_algebraic) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(offset * num_of_states) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(offset * num_of_states) + V]+20.0000)/10.0000));
ALGEBRAIC[(offset * num_of_algebraic) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(offset * num_of_states) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(offset * num_of_states) + V]+5.00000)/6.00000));
ALGEBRAIC[(offset * num_of_algebraic) + f] =  CONSTANTS[(offset * num_of_constants) + Aff]*STATES[(offset * num_of_states) + ff]+ CONSTANTS[(offset * num_of_constants) + Afs]*STATES[(offset * num_of_states) + fs];
ALGEBRAIC[(offset * num_of_algebraic) + fcass] = ALGEBRAIC[(offset * num_of_algebraic) + fss];
ALGEBRAIC[(offset * num_of_algebraic) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(offset * num_of_states) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(offset * num_of_states) + V] - 4.00000)/7.00000));
ALGEBRAIC[(offset * num_of_algebraic) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(offset * num_of_states) + V]/3.00000)+ 0.000120000*exp(STATES[(offset * num_of_states) + V]/7.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(offset * num_of_states) + V] - 10.0000)/10.0000));
ALGEBRAIC[(offset * num_of_algebraic) + Afcas] = 1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + Afcaf];
ALGEBRAIC[(offset * num_of_algebraic) + fca] =  ALGEBRAIC[(offset * num_of_algebraic) + Afcaf]*STATES[(offset * num_of_states) + fcaf]+ ALGEBRAIC[(offset * num_of_algebraic) + Afcas]*STATES[(offset * num_of_states) + fcas];
ALGEBRAIC[(offset * num_of_algebraic) + tffp] =  2.50000*ALGEBRAIC[(offset * num_of_algebraic) + tff];
ALGEBRAIC[(offset * num_of_algebraic) + fp] =  CONSTANTS[(offset * num_of_constants) + Aff]*STATES[(offset * num_of_states) + ffp]+ CONSTANTS[(offset * num_of_constants) + Afs]*STATES[(offset * num_of_states) + fs];
ALGEBRAIC[(offset * num_of_algebraic) + tfcafp] =  2.50000*ALGEBRAIC[(offset * num_of_algebraic) + tfcaf];
ALGEBRAIC[(offset * num_of_algebraic) + fcap] =  ALGEBRAIC[(offset * num_of_algebraic) + Afcaf]*STATES[(offset * num_of_states) + fcafp]+ ALGEBRAIC[(offset * num_of_algebraic) + Afcas]*STATES[(offset * num_of_states) + fcas];
ALGEBRAIC[(offset * num_of_algebraic) + km2n] =  STATES[(offset * num_of_states) + jca]*1.00000;
ALGEBRAIC[(offset * num_of_algebraic) + anca] = 1.00000/(CONSTANTS[(offset * num_of_constants) + k2n]/ALGEBRAIC[(offset * num_of_algebraic) + km2n]+pow(1.00000+CONSTANTS[(offset * num_of_constants) + Kmn]/STATES[(offset * num_of_states) + cass], 4.00000));
ALGEBRAIC[(offset * num_of_algebraic) + PhiCaL] = ( 4.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( STATES[(offset * num_of_states) + cass]*exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.341000*CONSTANTS[(offset * num_of_constants) + cao]))/(exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + PhiCaNa] = ( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( 0.750000*STATES[(offset * num_of_states) + nass]*exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.750000*CONSTANTS[(offset * num_of_constants) + nao]))/(exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + PhiCaK] = ( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( 0.750000*STATES[(offset * num_of_states) + kss]*exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.750000*CONSTANTS[(offset * num_of_constants) + ko]))/(exp( 1.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + ICaL] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fICaLp])*CONSTANTS[(offset * num_of_constants) + PCa]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaL]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + f]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fca]*STATES[(offset * num_of_states) + nca])+ ALGEBRAIC[(offset * num_of_algebraic) + fICaLp]*CONSTANTS[(offset * num_of_constants) + PCap]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaL]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + fp]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fcap]*STATES[(offset * num_of_states) + nca]);
ALGEBRAIC[(offset * num_of_algebraic) + ICaNa] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fICaLp])*CONSTANTS[(offset * num_of_constants) + PCaNa]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaNa]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + f]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fca]*STATES[(offset * num_of_states) + nca])+ ALGEBRAIC[(offset * num_of_algebraic) + fICaLp]*CONSTANTS[(offset * num_of_constants) + PCaNap]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaNa]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + fp]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fcap]*STATES[(offset * num_of_states) + nca]);
ALGEBRAIC[(offset * num_of_algebraic) + ICaK] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fICaLp])*CONSTANTS[(offset * num_of_constants) + PCaK]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaK]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + f]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fca]*STATES[(offset * num_of_states) + nca])+ ALGEBRAIC[(offset * num_of_algebraic) + fICaLp]*CONSTANTS[(offset * num_of_constants) + PCaKp]*ALGEBRAIC[(offset * num_of_algebraic) + PhiCaK]*STATES[(offset * num_of_states) + d]*( ALGEBRAIC[(offset * num_of_algebraic) + fp]*(1.00000 - STATES[(offset * num_of_states) + nca])+ STATES[(offset * num_of_states) + jca]*ALGEBRAIC[(offset * num_of_algebraic) + fcap]*STATES[(offset * num_of_states) + nca]);
ALGEBRAIC[(offset * num_of_algebraic) + ICab] = ( CONSTANTS[(offset * num_of_constants) + PCab]*4.00000*ALGEBRAIC[(offset * num_of_algebraic) + vffrt]*( STATES[(offset * num_of_states) + cai]*exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) -  0.341000*CONSTANTS[(offset * num_of_constants) + cao]))/(exp( 2.00000*ALGEBRAIC[(offset * num_of_algebraic) + vfrt]) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + IpCa] = ( CONSTANTS[(offset * num_of_constants) + GpCa]*STATES[(offset * num_of_states) + cai])/(CONSTANTS[(offset * num_of_constants) + KmCap]+STATES[(offset * num_of_states) + cai]);
ALGEBRAIC[(offset * num_of_algebraic) + hna] = exp(( CONSTANTS[(offset * num_of_constants) + qna]*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + hca] = exp(( CONSTANTS[(offset * num_of_constants) + qca]*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + h1_i] = 1.00000+ (STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h2_i] = ( STATES[(offset * num_of_states) + nai]*ALGEBRAIC[(offset * num_of_algebraic) + hna])/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + h1_i]);
ALGEBRAIC[(offset * num_of_algebraic) + h3_i] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h1_i];
ALGEBRAIC[(offset * num_of_algebraic) + h4_i] = 1.00000+ (STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + kna1])*(1.00000+STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + h5_i] = ( STATES[(offset * num_of_states) + nai]*STATES[(offset * num_of_states) + nai])/( ALGEBRAIC[(offset * num_of_algebraic) + h4_i]*CONSTANTS[(offset * num_of_constants) + kna1]*CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + h6_i] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h4_i];
ALGEBRAIC[(offset * num_of_algebraic) + h7_i] = 1.00000+ (CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+1.00000/ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h8_i] = CONSTANTS[(offset * num_of_constants) + nao]/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + hna]*ALGEBRAIC[(offset * num_of_algebraic) + h7_i]);
ALGEBRAIC[(offset * num_of_algebraic) + h9_i] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h7_i];
ALGEBRAIC[(offset * num_of_algebraic) + k3p_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h9_i]*CONSTANTS[(offset * num_of_constants) + wca];
ALGEBRAIC[(offset * num_of_algebraic) + k3pp_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_i]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + k3_i] = ALGEBRAIC[(offset * num_of_algebraic) + k3p_i]+ALGEBRAIC[(offset * num_of_algebraic) + k3pp_i];
ALGEBRAIC[(offset * num_of_algebraic) + k4p_i] = ( ALGEBRAIC[(offset * num_of_algebraic) + h3_i]*CONSTANTS[(offset * num_of_constants) + wca])/ALGEBRAIC[(offset * num_of_algebraic) + hca];
ALGEBRAIC[(offset * num_of_algebraic) + k4pp_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h2_i]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + k4_i] = ALGEBRAIC[(offset * num_of_algebraic) + k4p_i]+ALGEBRAIC[(offset * num_of_algebraic) + k4pp_i];
ALGEBRAIC[(offset * num_of_algebraic) + k6_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h6_i]*STATES[(offset * num_of_states) + cai]*CONSTANTS[(offset * num_of_constants) + kcaon];
ALGEBRAIC[(offset * num_of_algebraic) + k7_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h5_i]*ALGEBRAIC[(offset * num_of_algebraic) + h2_i]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + k8_i] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_i]*CONSTANTS[(offset * num_of_constants) + h11_i]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + x1_i] =  CONSTANTS[(offset * num_of_constants) + k2_i]*ALGEBRAIC[(offset * num_of_algebraic) + k4_i]*(ALGEBRAIC[(offset * num_of_algebraic) + k7_i]+ALGEBRAIC[(offset * num_of_algebraic) + k6_i])+ CONSTANTS[(offset * num_of_constants) + k5_i]*ALGEBRAIC[(offset * num_of_algebraic) + k7_i]*(CONSTANTS[(offset * num_of_constants) + k2_i]+ALGEBRAIC[(offset * num_of_algebraic) + k3_i]);
ALGEBRAIC[(offset * num_of_algebraic) + x2_i] =  CONSTANTS[(offset * num_of_constants) + k1_i]*ALGEBRAIC[(offset * num_of_algebraic) + k7_i]*(ALGEBRAIC[(offset * num_of_algebraic) + k4_i]+CONSTANTS[(offset * num_of_constants) + k5_i])+ ALGEBRAIC[(offset * num_of_algebraic) + k4_i]*ALGEBRAIC[(offset * num_of_algebraic) + k6_i]*(CONSTANTS[(offset * num_of_constants) + k1_i]+ALGEBRAIC[(offset * num_of_algebraic) + k8_i]);
ALGEBRAIC[(offset * num_of_algebraic) + x3_i] =  CONSTANTS[(offset * num_of_constants) + k1_i]*ALGEBRAIC[(offset * num_of_algebraic) + k3_i]*(ALGEBRAIC[(offset * num_of_algebraic) + k7_i]+ALGEBRAIC[(offset * num_of_algebraic) + k6_i])+ ALGEBRAIC[(offset * num_of_algebraic) + k8_i]*ALGEBRAIC[(offset * num_of_algebraic) + k6_i]*(CONSTANTS[(offset * num_of_constants) + k2_i]+ALGEBRAIC[(offset * num_of_algebraic) + k3_i]);
ALGEBRAIC[(offset * num_of_algebraic) + x4_i] =  CONSTANTS[(offset * num_of_constants) + k2_i]*ALGEBRAIC[(offset * num_of_algebraic) + k8_i]*(ALGEBRAIC[(offset * num_of_algebraic) + k4_i]+CONSTANTS[(offset * num_of_constants) + k5_i])+ ALGEBRAIC[(offset * num_of_algebraic) + k3_i]*CONSTANTS[(offset * num_of_constants) + k5_i]*(CONSTANTS[(offset * num_of_constants) + k1_i]+ALGEBRAIC[(offset * num_of_algebraic) + k8_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E1_i] = ALGEBRAIC[(offset * num_of_algebraic) + x1_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E2_i] = ALGEBRAIC[(offset * num_of_algebraic) + x2_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E3_i] = ALGEBRAIC[(offset * num_of_algebraic) + x3_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + E4_i] = ALGEBRAIC[(offset * num_of_algebraic) + x4_i]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_i]+ALGEBRAIC[(offset * num_of_algebraic) + x2_i]+ALGEBRAIC[(offset * num_of_algebraic) + x3_i]+ALGEBRAIC[(offset * num_of_algebraic) + x4_i]);
ALGEBRAIC[(offset * num_of_algebraic) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(offset * num_of_constants) + KmCaAct]/STATES[(offset * num_of_states) + cai], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_i] =  ALGEBRAIC[(offset * num_of_algebraic) + E2_i]*CONSTANTS[(offset * num_of_constants) + k2_i] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_i]*CONSTANTS[(offset * num_of_constants) + k1_i];
ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E4_i]*ALGEBRAIC[(offset * num_of_algebraic) + k7_i] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_i]*ALGEBRAIC[(offset * num_of_algebraic) + k8_i])+ ALGEBRAIC[(offset * num_of_algebraic) + E3_i]*ALGEBRAIC[(offset * num_of_algebraic) + k4pp_i]) -  ALGEBRAIC[(offset * num_of_algebraic) + E2_i]*ALGEBRAIC[(offset * num_of_algebraic) + k3pp_i];
ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i] =  0.800000*CONSTANTS[(offset * num_of_constants) + Gncx]*ALGEBRAIC[(offset * num_of_algebraic) + allo_i]*( CONSTANTS[(offset * num_of_constants) + zna]*ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_i]+ CONSTANTS[(offset * num_of_constants) + zca]*ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_i]);
ALGEBRAIC[(offset * num_of_algebraic) + h1_ss] = 1.00000+ (STATES[(offset * num_of_states) + nass]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h2_ss] = ( STATES[(offset * num_of_states) + nass]*ALGEBRAIC[(offset * num_of_algebraic) + hna])/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + h1_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + h3_ss] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h1_ss];
ALGEBRAIC[(offset * num_of_algebraic) + h4_ss] = 1.00000+ (STATES[(offset * num_of_states) + nass]/CONSTANTS[(offset * num_of_constants) + kna1])*(1.00000+STATES[(offset * num_of_states) + nass]/CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + h5_ss] = ( STATES[(offset * num_of_states) + nass]*STATES[(offset * num_of_states) + nass])/( ALGEBRAIC[(offset * num_of_algebraic) + h4_ss]*CONSTANTS[(offset * num_of_constants) + kna1]*CONSTANTS[(offset * num_of_constants) + kna2]);
ALGEBRAIC[(offset * num_of_algebraic) + h6_ss] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h4_ss];
ALGEBRAIC[(offset * num_of_algebraic) + h7_ss] = 1.00000+ (CONSTANTS[(offset * num_of_constants) + nao]/CONSTANTS[(offset * num_of_constants) + kna3])*(1.00000+1.00000/ALGEBRAIC[(offset * num_of_algebraic) + hna]);
ALGEBRAIC[(offset * num_of_algebraic) + h8_ss] = CONSTANTS[(offset * num_of_constants) + nao]/( CONSTANTS[(offset * num_of_constants) + kna3]*ALGEBRAIC[(offset * num_of_algebraic) + hna]*ALGEBRAIC[(offset * num_of_algebraic) + h7_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + h9_ss] = 1.00000/ALGEBRAIC[(offset * num_of_algebraic) + h7_ss];
ALGEBRAIC[(offset * num_of_algebraic) + k3p_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h9_ss]*CONSTANTS[(offset * num_of_constants) + wca];
ALGEBRAIC[(offset * num_of_algebraic) + k3pp_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_ss]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + k3_ss] = ALGEBRAIC[(offset * num_of_algebraic) + k3p_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k3pp_ss];
ALGEBRAIC[(offset * num_of_algebraic) + k4p_ss] = ( ALGEBRAIC[(offset * num_of_algebraic) + h3_ss]*CONSTANTS[(offset * num_of_constants) + wca])/ALGEBRAIC[(offset * num_of_algebraic) + hca];
ALGEBRAIC[(offset * num_of_algebraic) + k4pp_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h2_ss]*CONSTANTS[(offset * num_of_constants) + wnaca];
ALGEBRAIC[(offset * num_of_algebraic) + k4_ss] = ALGEBRAIC[(offset * num_of_algebraic) + k4p_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k4pp_ss];
ALGEBRAIC[(offset * num_of_algebraic) + k6_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h6_ss]*STATES[(offset * num_of_states) + cass]*CONSTANTS[(offset * num_of_constants) + kcaon];
ALGEBRAIC[(offset * num_of_algebraic) + k7_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h5_ss]*ALGEBRAIC[(offset * num_of_algebraic) + h2_ss]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + k8_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + h8_ss]*CONSTANTS[(offset * num_of_constants) + h11_ss]*CONSTANTS[(offset * num_of_constants) + wna];
ALGEBRAIC[(offset * num_of_algebraic) + x1_ss] =  CONSTANTS[(offset * num_of_constants) + k2_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k6_ss])+ CONSTANTS[(offset * num_of_constants) + k5_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]*(CONSTANTS[(offset * num_of_constants) + k2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + x2_ss] =  CONSTANTS[(offset * num_of_constants) + k1_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]+CONSTANTS[(offset * num_of_constants) + k5_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k6_ss]*(CONSTANTS[(offset * num_of_constants) + k1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + x3_ss] =  CONSTANTS[(offset * num_of_constants) + k1_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k7_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k6_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k6_ss]*(CONSTANTS[(offset * num_of_constants) + k2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + x4_ss] =  CONSTANTS[(offset * num_of_constants) + k2_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]*(ALGEBRAIC[(offset * num_of_algebraic) + k4_ss]+CONSTANTS[(offset * num_of_constants) + k5_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + k3_ss]*CONSTANTS[(offset * num_of_constants) + k5_ss]*(CONSTANTS[(offset * num_of_constants) + k1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + k8_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E1_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E2_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E3_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + E4_ss] = ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]/(ALGEBRAIC[(offset * num_of_algebraic) + x1_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x2_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x3_ss]+ALGEBRAIC[(offset * num_of_algebraic) + x4_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(offset * num_of_constants) + KmCaAct]/STATES[(offset * num_of_states) + cass], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_ss] =  ALGEBRAIC[(offset * num_of_algebraic) + E2_ss]*CONSTANTS[(offset * num_of_constants) + k2_ss] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_ss]*CONSTANTS[(offset * num_of_constants) + k1_ss];
ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E4_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k7_ss] -  ALGEBRAIC[(offset * num_of_algebraic) + E1_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k8_ss])+ ALGEBRAIC[(offset * num_of_algebraic) + E3_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k4pp_ss]) -  ALGEBRAIC[(offset * num_of_algebraic) + E2_ss]*ALGEBRAIC[(offset * num_of_algebraic) + k3pp_ss];
ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss] =  0.200000*CONSTANTS[(offset * num_of_constants) + Gncx]*ALGEBRAIC[(offset * num_of_algebraic) + allo_ss]*( CONSTANTS[(offset * num_of_constants) + zna]*ALGEBRAIC[(offset * num_of_algebraic) + JncxNa_ss]+ CONSTANTS[(offset * num_of_constants) + zca]*ALGEBRAIC[(offset * num_of_algebraic) + JncxCa_ss]);
ALGEBRAIC[(offset * num_of_algebraic) + Knai] =  CONSTANTS[(offset * num_of_constants) + Knai0]*exp(( CONSTANTS[(offset * num_of_constants) + delta]*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( 3.00000*CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + Knao] =  CONSTANTS[(offset * num_of_constants) + Knao0]*exp(( (1.00000 - CONSTANTS[(offset * num_of_constants) + delta])*STATES[(offset * num_of_states) + V]*CONSTANTS[(offset * num_of_constants) + F])/( 3.00000*CONSTANTS[(offset * num_of_constants) + R]*CONSTANTS[(offset * num_of_constants) + T]));
ALGEBRAIC[(offset * num_of_algebraic) + P] = CONSTANTS[(offset * num_of_constants) + eP]/(1.00000+CONSTANTS[(offset * num_of_constants) + H]/CONSTANTS[(offset * num_of_constants) + Khp]+STATES[(offset * num_of_states) + nai]/CONSTANTS[(offset * num_of_constants) + Knap]+STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kxkur]);
ALGEBRAIC[(offset * num_of_algebraic) + a1] = ( CONSTANTS[(offset * num_of_constants) + k1p]*pow(STATES[(offset * num_of_states) + nai]/ALGEBRAIC[(offset * num_of_algebraic) + Knai], 3.00000))/((pow(1.00000+STATES[(offset * num_of_states) + nai]/ALGEBRAIC[(offset * num_of_algebraic) + Knai], 3.00000)+pow(1.00000+STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + a3] = ( CONSTANTS[(offset * num_of_constants) + k3p]*pow(CONSTANTS[(offset * num_of_constants) + ko]/CONSTANTS[(offset * num_of_constants) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(offset * num_of_constants) + nao]/ALGEBRAIC[(offset * num_of_algebraic) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(offset * num_of_constants) + ko]/CONSTANTS[(offset * num_of_constants) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + b2] = ( CONSTANTS[(offset * num_of_constants) + k2m]*pow(CONSTANTS[(offset * num_of_constants) + nao]/ALGEBRAIC[(offset * num_of_algebraic) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(offset * num_of_constants) + nao]/ALGEBRAIC[(offset * num_of_algebraic) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(offset * num_of_constants) + ko]/CONSTANTS[(offset * num_of_constants) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + b3] = ( CONSTANTS[(offset * num_of_constants) + k3m]*ALGEBRAIC[(offset * num_of_algebraic) + P]*CONSTANTS[(offset * num_of_constants) + H])/(1.00000+CONSTANTS[(offset * num_of_constants) + MgATP]/CONSTANTS[(offset * num_of_constants) + Kmgatp]);
ALGEBRAIC[(offset * num_of_algebraic) + b4] = ( CONSTANTS[(offset * num_of_constants) + k4m]*pow(STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kki], 2.00000))/((pow(1.00000+STATES[(offset * num_of_states) + nai]/ALGEBRAIC[(offset * num_of_algebraic) + Knai], 3.00000)+pow(1.00000+STATES[(offset * num_of_states) + ki]/CONSTANTS[(offset * num_of_constants) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(offset * num_of_algebraic) + x1] =  CONSTANTS[(offset * num_of_constants) + a4]*ALGEBRAIC[(offset * num_of_algebraic) + a1]*CONSTANTS[(offset * num_of_constants) + a2]+ ALGEBRAIC[(offset * num_of_algebraic) + b2]*ALGEBRAIC[(offset * num_of_algebraic) + b4]*ALGEBRAIC[(offset * num_of_algebraic) + b3]+ CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + b4]*ALGEBRAIC[(offset * num_of_algebraic) + b3]+ ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + a1]*CONSTANTS[(offset * num_of_constants) + a2];
ALGEBRAIC[(offset * num_of_algebraic) + x2] =  ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + b1]*ALGEBRAIC[(offset * num_of_algebraic) + b4]+ ALGEBRAIC[(offset * num_of_algebraic) + a1]*CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + a3]+ ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + b1]*ALGEBRAIC[(offset * num_of_algebraic) + b4]+ CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + a3]*ALGEBRAIC[(offset * num_of_algebraic) + b4];
ALGEBRAIC[(offset * num_of_algebraic) + x3] =  CONSTANTS[(offset * num_of_constants) + a2]*ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + a4]+ ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + b1]+ ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + b1]*CONSTANTS[(offset * num_of_constants) + a4]+ ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + a4]*CONSTANTS[(offset * num_of_constants) + b1];
ALGEBRAIC[(offset * num_of_algebraic) + x4] =  ALGEBRAIC[(offset * num_of_algebraic) + b4]*ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + b2]+ ALGEBRAIC[(offset * num_of_algebraic) + a3]*CONSTANTS[(offset * num_of_constants) + a4]*ALGEBRAIC[(offset * num_of_algebraic) + a1]+ ALGEBRAIC[(offset * num_of_algebraic) + b2]*CONSTANTS[(offset * num_of_constants) + a4]*ALGEBRAIC[(offset * num_of_algebraic) + a1]+ ALGEBRAIC[(offset * num_of_algebraic) + b3]*ALGEBRAIC[(offset * num_of_algebraic) + b2]*ALGEBRAIC[(offset * num_of_algebraic) + a1];
ALGEBRAIC[(offset * num_of_algebraic) + E1] = ALGEBRAIC[(offset * num_of_algebraic) + x1]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + E2] = ALGEBRAIC[(offset * num_of_algebraic) + x2]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + E4] = ALGEBRAIC[(offset * num_of_algebraic) + x4]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + E3] = ALGEBRAIC[(offset * num_of_algebraic) + x3]/(ALGEBRAIC[(offset * num_of_algebraic) + x1]+ALGEBRAIC[(offset * num_of_algebraic) + x2]+ALGEBRAIC[(offset * num_of_algebraic) + x3]+ALGEBRAIC[(offset * num_of_algebraic) + x4]);
ALGEBRAIC[(offset * num_of_algebraic) + JnakNa] =  3.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E1]*ALGEBRAIC[(offset * num_of_algebraic) + a3] -  ALGEBRAIC[(offset * num_of_algebraic) + E2]*ALGEBRAIC[(offset * num_of_algebraic) + b3]);
ALGEBRAIC[(offset * num_of_algebraic) + JnakK] =  2.00000*( ALGEBRAIC[(offset * num_of_algebraic) + E4]*CONSTANTS[(offset * num_of_constants) + b1] -  ALGEBRAIC[(offset * num_of_algebraic) + E3]*ALGEBRAIC[(offset * num_of_algebraic) + a1]);
ALGEBRAIC[(offset * num_of_algebraic) + INaK] =  CONSTANTS[(offset * num_of_constants) + Pnak]*( CONSTANTS[(offset * num_of_constants) + zna]*ALGEBRAIC[(offset * num_of_algebraic) + JnakNa]+ CONSTANTS[(offset * num_of_constants) + zk]*ALGEBRAIC[(offset * num_of_algebraic) + JnakK]);
ALGEBRAIC[(offset * num_of_algebraic) + Jdiff] = (STATES[(offset * num_of_states) + cass] - STATES[(offset * num_of_states) + cai])/0.200000;
ALGEBRAIC[(offset * num_of_algebraic) + JdiffK] = (STATES[(offset * num_of_states) + kss] - STATES[(offset * num_of_states) + ki])/2.00000;
ALGEBRAIC[(offset * num_of_algebraic) + JdiffNa] = (STATES[(offset * num_of_states) + nass] - STATES[(offset * num_of_states) + nai])/2.00000;
ALGEBRAIC[(offset * num_of_algebraic) + Jupnp] = ( CONSTANTS[(offset * num_of_constants) + upScale]*0.00437500*STATES[(offset * num_of_states) + cai])/(STATES[(offset * num_of_states) + cai]+0.000920000);
ALGEBRAIC[(offset * num_of_algebraic) + Jupp] = ( CONSTANTS[(offset * num_of_constants) + upScale]*2.75000*0.00437500*STATES[(offset * num_of_states) + cai])/((STATES[(offset * num_of_states) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(offset * num_of_algebraic) + fJupp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + Jleak] = ( 0.00393750*STATES[(offset * num_of_states) + cansr])/15.0000;
ALGEBRAIC[(offset * num_of_algebraic) + Jup] = ( (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fJupp])*ALGEBRAIC[(offset * num_of_algebraic) + Jupnp]+ ALGEBRAIC[(offset * num_of_algebraic) + fJupp]*ALGEBRAIC[(offset * num_of_algebraic) + Jupp]) - ALGEBRAIC[(offset * num_of_algebraic) + Jleak];
ALGEBRAIC[(offset * num_of_algebraic) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(offset * num_of_constants) + KmCaMK]/ALGEBRAIC[(offset * num_of_algebraic) + CaMKa]);
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf_temp] = ( CONSTANTS[(offset * num_of_constants) + a_rel]*- ALGEBRAIC[(offset * num_of_algebraic) + ICaL])/(1.00000+ 1.00000*pow(1.50000/STATES[(offset * num_of_states) + cajsr], 8.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] = (CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf_temp]*1.70000 : ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_relp_temp] = CONSTANTS[(offset * num_of_constants) + bt]/(1.00000+0.0123000/STATES[(offset * num_of_states) + cajsr]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_rel] = (ALGEBRAIC[(offset * num_of_algebraic) + tau_rel_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(offset * num_of_algebraic) + tau_rel_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_temp] = ( CONSTANTS[(offset * num_of_constants) + a_relp]*- ALGEBRAIC[(offset * num_of_algebraic) + ICaL])/(1.00000+pow(1.50000/STATES[(offset * num_of_states) + cajsr], 8.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] = (CONSTANTS[(offset * num_of_constants) + celltype]==2.00000 ?  ALGEBRAIC[(offset * num_of_algebraic) + Jrel_temp]*1.70000 : ALGEBRAIC[(offset * num_of_algebraic) + Jrel_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_rel_temp] = CONSTANTS[(offset * num_of_constants) + bt]/(1.00000+0.0123000/STATES[(offset * num_of_states) + cajsr]);
ALGEBRAIC[(offset * num_of_algebraic) + tau_relp] = (ALGEBRAIC[(offset * num_of_algebraic) + tau_relp_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(offset * num_of_algebraic) + tau_relp_temp]);
ALGEBRAIC[(offset * num_of_algebraic) + Jrel] =  (1.00000 - ALGEBRAIC[(offset * num_of_algebraic) + fJrelp])*STATES[(offset * num_of_states) + Jrelnp]+ ALGEBRAIC[(offset * num_of_algebraic) + fJrelp]*STATES[(offset * num_of_states) + Jrelp];
ALGEBRAIC[(offset * num_of_algebraic) + Jtr] = (STATES[(offset * num_of_states) + cansr] - STATES[(offset * num_of_states) + cajsr])/100.000;
ALGEBRAIC[(offset * num_of_algebraic) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(offset * num_of_constants) + cmdnmax]*CONSTANTS[(offset * num_of_constants) + kmcmdn])/pow(CONSTANTS[(offset * num_of_constants) + kmcmdn]+STATES[(offset * num_of_states) + cai], 2.00000)+( CONSTANTS[(offset * num_of_constants) + trpnmax]*CONSTANTS[(offset * num_of_constants) + kmtrpn])/pow(CONSTANTS[(offset * num_of_constants) + kmtrpn]+STATES[(offset * num_of_states) + cai], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(offset * num_of_constants) + BSRmax]*CONSTANTS[(offset * num_of_constants) + KmBSR])/pow(CONSTANTS[(offset * num_of_constants) + KmBSR]+STATES[(offset * num_of_states) + cass], 2.00000)+( CONSTANTS[(offset * num_of_constants) + BSLmax]*CONSTANTS[(offset * num_of_constants) + KmBSL])/pow(CONSTANTS[(offset * num_of_constants) + KmBSL]+STATES[(offset * num_of_states) + cass], 2.00000));
ALGEBRAIC[(offset * num_of_algebraic) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(offset * num_of_constants) + csqnmax]*CONSTANTS[(offset * num_of_constants) + kmcsqn])/pow(CONSTANTS[(offset * num_of_constants) + kmcsqn]+STATES[(offset * num_of_states) + cajsr], 2.00000));
RATES[(offset * num_of_rates) + m] = (ALGEBRAIC[(offset * num_of_algebraic) + mss] - STATES[(offset * num_of_states) + m])/ALGEBRAIC[(offset * num_of_algebraic) + tm];
RATES[(offset * num_of_rates) + j] = (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + j])/ALGEBRAIC[(offset * num_of_algebraic) + tj];
RATES[(offset * num_of_rates) + jp] = (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + jp])/ALGEBRAIC[(offset * num_of_algebraic) + tjp];
RATES[(offset * num_of_rates) + hf] = (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hf])/ALGEBRAIC[(offset * num_of_algebraic) + thf];
RATES[(offset * num_of_rates) + hs] = (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hs])/ALGEBRAIC[(offset * num_of_algebraic) + ths];
RATES[(offset * num_of_rates) + hsp] = (ALGEBRAIC[(offset * num_of_algebraic) + hssp] - STATES[(offset * num_of_states) + hsp])/ALGEBRAIC[(offset * num_of_algebraic) + thsp];
RATES[(offset * num_of_rates) + mL] = (ALGEBRAIC[(offset * num_of_algebraic) + mLss] - STATES[(offset * num_of_states) + mL])/ALGEBRAIC[(offset * num_of_algebraic) + tmL];
RATES[(offset * num_of_rates) + hL] = (ALGEBRAIC[(offset * num_of_algebraic) + hLss] - STATES[(offset * num_of_states) + hL])/CONSTANTS[(offset * num_of_constants) + thL];
RATES[(offset * num_of_rates) + hLp] = (ALGEBRAIC[(offset * num_of_algebraic) + hLssp] - STATES[(offset * num_of_states) + hLp])/CONSTANTS[(offset * num_of_constants) + thLp];
RATES[(offset * num_of_rates) + a] = (ALGEBRAIC[(offset * num_of_algebraic) + ass] - STATES[(offset * num_of_states) + a])/ALGEBRAIC[(offset * num_of_algebraic) + ta];
RATES[(offset * num_of_rates) + ap] = (ALGEBRAIC[(offset * num_of_algebraic) + assp] - STATES[(offset * num_of_states) + ap])/ALGEBRAIC[(offset * num_of_algebraic) + ta];
RATES[(offset * num_of_rates) + iF] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iF])/ALGEBRAIC[(offset * num_of_algebraic) + tiF];
RATES[(offset * num_of_rates) + iS] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iS])/ALGEBRAIC[(offset * num_of_algebraic) + tiS];
RATES[(offset * num_of_rates) + iFp] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iFp])/ALGEBRAIC[(offset * num_of_algebraic) + tiFp];
RATES[(offset * num_of_rates) + iSp] = (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iSp])/ALGEBRAIC[(offset * num_of_algebraic) + tiSp];
RATES[(offset * num_of_rates) + xrf] = (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrf])/ALGEBRAIC[(offset * num_of_algebraic) + txrf];
RATES[(offset * num_of_rates) + xrs] = (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrs])/ALGEBRAIC[(offset * num_of_algebraic) + txrs];
RATES[(offset * num_of_rates) + xs1] = (ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] - STATES[(offset * num_of_states) + xs1])/ALGEBRAIC[(offset * num_of_algebraic) + txs1];
RATES[(offset * num_of_rates) + xs2] = (ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] - STATES[(offset * num_of_states) + xs2])/ALGEBRAIC[(offset * num_of_algebraic) + txs2];
RATES[(offset * num_of_rates) + xk1] = (ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] - STATES[(offset * num_of_states) + xk1])/ALGEBRAIC[(offset * num_of_algebraic) + txk1];
RATES[(offset * num_of_rates) + d] = (ALGEBRAIC[(offset * num_of_algebraic) + dss] - STATES[(offset * num_of_states) + d])/ALGEBRAIC[(offset * num_of_algebraic) + td];
RATES[(offset * num_of_rates) + ff] = (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ff])/ALGEBRAIC[(offset * num_of_algebraic) + tff];
RATES[(offset * num_of_rates) + fs] = (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + fs])/ALGEBRAIC[(offset * num_of_algebraic) + tfs];
RATES[(offset * num_of_rates) + fcaf] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcaf])/ALGEBRAIC[(offset * num_of_algebraic) + tfcaf];
RATES[(offset * num_of_rates) + nca] =  ALGEBRAIC[(offset * num_of_algebraic) + anca]*CONSTANTS[(offset * num_of_constants) + k2n] -  STATES[(offset * num_of_states) + nca]*ALGEBRAIC[(offset * num_of_algebraic) + km2n];
RATES[(offset * num_of_rates) + jca] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + jca])/CONSTANTS[(offset * num_of_constants) + tjca];
RATES[(offset * num_of_rates) + fcas] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcas])/ALGEBRAIC[(offset * num_of_algebraic) + tfcas];
RATES[(offset * num_of_rates) + ffp] = (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ffp])/ALGEBRAIC[(offset * num_of_algebraic) + tffp];
RATES[(offset * num_of_rates) + fcafp] = (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcafp])/ALGEBRAIC[(offset * num_of_algebraic) + tfcafp];
RATES[(offset * num_of_rates) + Jrelnp] = (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] - STATES[(offset * num_of_states) + Jrelnp])/ALGEBRAIC[(offset * num_of_algebraic) + tau_rel];
RATES[(offset * num_of_rates) + Jrelp] = (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] - STATES[(offset * num_of_states) + Jrelp])/ALGEBRAIC[(offset * num_of_algebraic) + tau_relp];
RATES[(offset * num_of_rates) + CaMKt] =  CONSTANTS[(offset * num_of_constants) + aCaMK]*ALGEBRAIC[(offset * num_of_algebraic) + CaMKb]*(ALGEBRAIC[(offset * num_of_algebraic) + CaMKb]+STATES[(offset * num_of_states) + CaMKt]) -  CONSTANTS[(offset * num_of_constants) + bCaMK]*STATES[(offset * num_of_states) + CaMKt];

RATES[(offset * num_of_rates) + nai] = ( - (ALGEBRAIC[(offset * num_of_algebraic) + INa]+ALGEBRAIC[(offset * num_of_algebraic) + INaL]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaK]+ALGEBRAIC[(offset * num_of_algebraic) + INab])*CONSTANTS[(offset * num_of_constants) + Acap]*CONSTANTS[(offset * num_of_constants) + cm])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vmyo])+( ALGEBRAIC[(offset * num_of_algebraic) + JdiffNa]*CONSTANTS[(offset * num_of_constants) + vss])/CONSTANTS[(offset * num_of_constants) + vmyo];

RATES[(offset * num_of_rates) + nass] = ( - (ALGEBRAIC[(offset * num_of_algebraic) + ICaNa]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vss]) - ALGEBRAIC[(offset * num_of_algebraic) + JdiffNa];
RATES[(offset * num_of_rates) + ki] = ( - ((ALGEBRAIC[(offset * num_of_algebraic) + Ito]+ALGEBRAIC[(offset * num_of_algebraic) + IKr]+ALGEBRAIC[(offset * num_of_algebraic) + IKs]+ALGEBRAIC[(offset * num_of_algebraic) + IK1]+ALGEBRAIC[(offset * num_of_algebraic) + IKb]+ALGEBRAIC[(offset * num_of_algebraic) + Istim]) -  2.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaK])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vmyo])+( ALGEBRAIC[(offset * num_of_algebraic) + JdiffK]*CONSTANTS[(offset * num_of_constants) + vss])/CONSTANTS[(offset * num_of_constants) + vmyo];
RATES[(offset * num_of_rates) + kss] = ( - ALGEBRAIC[(offset * num_of_algebraic) + ICaK]*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vss]) - ALGEBRAIC[(offset * num_of_algebraic) + JdiffK];
RATES[(offset * num_of_rates) + cai] =  ALGEBRAIC[(offset * num_of_algebraic) + Bcai]*((( - ((ALGEBRAIC[(offset * num_of_algebraic) + IpCa]+ALGEBRAIC[(offset * num_of_algebraic) + ICab]) -  2.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( 2.00000*CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vmyo]) - ( ALGEBRAIC[(offset * num_of_algebraic) + Jup]*CONSTANTS[(offset * num_of_constants) + vnsr])/CONSTANTS[(offset * num_of_constants) + vmyo])+( ALGEBRAIC[(offset * num_of_algebraic) + Jdiff]*CONSTANTS[(offset * num_of_constants) + vss])/CONSTANTS[(offset * num_of_constants) + vmyo]);
RATES[(offset * num_of_rates) + cass] =  ALGEBRAIC[(offset * num_of_algebraic) + Bcass]*((( - (ALGEBRAIC[(offset * num_of_algebraic) + ICaL] -  2.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss])*CONSTANTS[(offset * num_of_constants) + cm]*CONSTANTS[(offset * num_of_constants) + Acap])/( 2.00000*CONSTANTS[(offset * num_of_constants) + F]*CONSTANTS[(offset * num_of_constants) + vss])+( ALGEBRAIC[(offset * num_of_algebraic) + Jrel]*CONSTANTS[(offset * num_of_constants) + vjsr])/CONSTANTS[(offset * num_of_constants) + vss]) - ALGEBRAIC[(offset * num_of_algebraic) + Jdiff]);
RATES[(offset * num_of_rates) + cansr] = ALGEBRAIC[(offset * num_of_algebraic) + Jup] - ( ALGEBRAIC[(offset * num_of_algebraic) + Jtr]*CONSTANTS[(offset * num_of_constants) + vjsr])/CONSTANTS[(offset * num_of_constants) + vnsr];
RATES[(offset * num_of_rates) + cajsr] =  ALGEBRAIC[(offset * num_of_algebraic) + Bcajsr]*(ALGEBRAIC[(offset * num_of_algebraic) + Jtr] - ALGEBRAIC[(offset * num_of_algebraic) + Jrel]);
RATES[(offset * num_of_rates) + V] = - (ALGEBRAIC[(offset * num_of_algebraic) + INa]+ALGEBRAIC[(offset * num_of_algebraic) + INaL]+ALGEBRAIC[(offset * num_of_algebraic) + Ito]+ALGEBRAIC[(offset * num_of_algebraic) + ICaL]+ALGEBRAIC[(offset * num_of_algebraic) + ICaNa]+ALGEBRAIC[(offset * num_of_algebraic) + ICaK]+ALGEBRAIC[(offset * num_of_algebraic) + IKr]+ALGEBRAIC[(offset * num_of_algebraic) + IKs]+ALGEBRAIC[(offset * num_of_algebraic) + IK1]+ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]+ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss]+ALGEBRAIC[(offset * num_of_algebraic) + INaK]+ALGEBRAIC[(offset * num_of_algebraic) + INab]+ALGEBRAIC[(offset * num_of_algebraic) + IKb]+ALGEBRAIC[(offset * num_of_algebraic) + IpCa]+ALGEBRAIC[(offset * num_of_algebraic) + ICab]+ALGEBRAIC[(offset * num_of_algebraic) + Istim]);
// for (int ZZZ = 0 ; ZZZ < num_of_rates; ZZZ++){
  // printf("core %d, nilai pembilang nai = %lf\n", offset, - (ALGEBRAIC[(offset * num_of_algebraic) + INa]+ALGEBRAIC[(offset * num_of_algebraic) + INaL]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]+ 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaK]+ALGEBRAIC[(offset * num_of_algebraic) + INab])*CONSTANTS[(offset * num_of_constants) + Acap]*CONSTANTS[(offset * num_of_constants) + cm]);
  // printf("core %d, nilai pembilang INa = %lf\n", offset, ALGEBRAIC[(offset * num_of_algebraic) + INa]); //nan
  // printf("core %d, nilai pembilang INaL = %lf\n", offset, ALGEBRAIC[(offset * num_of_algebraic) + INaL]); //nan
  // printf("core %d, nilai pembilang INaCa_i = %lf\n", offset, 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]);
  // printf("core %d, nilai pembilang INaK = %lf\n", offset, 3.00000*ALGEBRAIC[(offset * num_of_algebraic) + INaK]);
  // printf("core %d, nilai pembilang INab = %lf\n", offset, ALGEBRAIC[(offset * num_of_algebraic) + INab]);
  // printf("core %d, nilai pembilang acap = %lf\n", offset, CONSTANTS[(offset * num_of_constants) + Acap]);
  // printf("core %d, nilai pembilang cm = %lf\n", offset, CONSTANTS[(offset * num_of_constants) + cm]);
  // printf("core %d, nilai states m : %lf\n", offset, STATES[m + (offset * num_of_states)]);
  // printf("core %d, nilai algebraic[tm] : %lf\n", offset, ALGEBRAIC[tm + (offset * num_of_algebraic)]);
  // // ( CONSTANTS[mtD1 + (offset * num_of_constants)]*exp((STATES[V + (offset * num_of_states)]+CONSTANTS[mtV1 + (offset * num_of_constants)])/CONSTANTS[mtV2 + (offset * num_of_constants)])+ CONSTANTS[mtD2 + (offset * num_of_constants) ]*exp(- (STATES[V + (offset * num_of_states) ]+CONSTANTS[mtV3 + (offset * num_of_constants)])/CONSTANTS[mtV4 + (offset * num_of_constants)]))
  // printf("core %d, nilai constants[mtD1] : %lf\n", offset, CONSTANTS[mtD1 + (offset * num_of_constants)]);
  // printf("core %d, nilai constants[mtV2] : %lf\n", offset, CONSTANTS[mtV2 + (offset * num_of_constants)]);
  // printf("core %d, nilai constants[mtD2] : %lf\n", offset, CONSTANTS[mtD2 + (offset * num_of_constants)]);
  // printf("core %d, nilai constants[mtV4] : %lf\n", offset, CONSTANTS[mtV4 + (offset * num_of_constants)]);
  // printf("core %d, nilai constants[mtV1] : %lf\n", offset, CONSTANTS[mtV1 + (offset * num_of_constants)]);
  // printf("core %d, nilai pembilang tm  : %lf\n", offset, (STATES[V + (offset * num_of_states)]+CONSTANTS[mtV1 + (offset * num_of_constants)]));
  // printf("core %d, nilai exp pertama : %lf\n", offset, (STATES[V + (offset * num_of_states)]+CONSTANTS[mtV1 + (offset * num_of_constants)])  /  CONSTANTS[mtV2 + (offset * num_of_constants)]  );
  // printf("core %d, nilai rates[v] : %lf\n", offset, - (ALGEBRAIC[(offset * num_of_algebraic) + INa]+ALGEBRAIC[(offset * num_of_algebraic) + INaL]+ALGEBRAIC[(offset * num_of_algebraic) + Ito]+ALGEBRAIC[(offset * num_of_algebraic) + ICaL]+ALGEBRAIC[(offset * num_of_algebraic) + ICaNa]+ALGEBRAIC[(offset * num_of_algebraic) + ICaK]+ALGEBRAIC[(offset * num_of_algebraic) + IKr]+ALGEBRAIC[(offset * num_of_algebraic) + IKs]+ALGEBRAIC[(offset * num_of_algebraic) + IK1]+ALGEBRAIC[(offset * num_of_algebraic) + INaCa_i]+ALGEBRAIC[(offset * num_of_algebraic) + INaCa_ss]+ALGEBRAIC[(offset * num_of_algebraic) + INaK]+ALGEBRAIC[(offset * num_of_algebraic) + INab]+ALGEBRAIC[(offset * num_of_algebraic) + IKb]+ALGEBRAIC[(offset * num_of_algebraic) + IpCa]+ALGEBRAIC[(offset * num_of_algebraic) + ICab]+ALGEBRAIC[(offset * num_of_algebraic) + Istim]));
  // // rates v NaN karena INa dan INaL bernilai NaN juga, jadi mau operasi pertambahan pun, hasilnya bakal nan. 

  // printf("core %d, nilai INa = %lf\n", offset, ALGEBRAIC[(offset * num_of_algebraic) + INa]); //nan
  // printf("core %d, nilai INaL = %lf\n", offset, ALGEBRAIC[(offset * num_of_algebraic) + INaL]); //nan
  // // ALGEBRAIC[INaL + (offset * num_of_algebraic) ] =  CONSTANTS[GNaL + (offset * num_of_constants) ]*(STATES[V + (offset * num_of_states)] - ALGEBRAIC[ENa + (offset * num_of_algebraic)])*STATES[mL + (offset * num_of_states) ]*( (1.00000 - ALGEBRAIC[fINaLp + (offset * num_of_algebraic)])*STATES[hL + (offset * num_of_states) ]+ ALGEBRAIC[fINaLp + (offset * num_of_algebraic) ]*STATES[hLp + (offset * num_of_states) ]);
  // printf("core %d, nilai Gnal = %lf\n", offset, CONSTANTS[GNaL + (offset * num_of_constants)] );
  // printf("core %d, nilai ENa = %lf\n", offset, ALGEBRAIC[ENa + (offset * num_of_algebraic)] ); //inf atau -nan
  // ALGEBRAIC[ENa + (offset * num_of_algebraic) ] =  (( CONSTANTS[R + (offset * num_of_constants) ]*CONSTANTS[T + (offset * num_of_constants) ])/CONSTANTS[F + (offset * num_of_constants) ])*log(CONSTANTS[nao + (offset * num_of_constants)]/STATES[nai + (offset * num_of_constants)]);
  // printf("core %d, nilai pembilang ENa = %lf\n", offset, ( CONSTANTS[R + (offset * num_of_constants) ]*CONSTANTS[T + (offset * num_of_constants) ]) );
  // printf("core %d, nilai penyebut ENa = %lf\n", offset, CONSTANTS[F + (offset * num_of_constants) ])*log( CONSTANTS[nao + (offset * num_of_constants)] / STATES[nai + (offset * num_of_constants)] );
  // printf("core %d, uraian ENa = (( %lf * %lf )/ %lf )*log(%lf / %lf );\n", offset, CONSTANTS[R + (offset * num_of_constants) ], CONSTANTS[T + (offset * num_of_constants) ], CONSTANTS[F + (offset * num_of_constants) ], CONSTANTS[nao + (offset * num_of_constants)], STATES[nai + (offset * num_of_constants)] );
 
  // printf("core %d, nilai mL = %lf\n", offset,STATES[mL + (offset * num_of_states) ]); //nan 
  // printf("core %d, mLss : %lf\n", offset,  1.00000/(1.00000+exp(- (STATES[V + (offset * num_of_states)]+42.8500)/5.26400)));
  // printf("core %d, update mL  = (%lf - %lf)/ %lf\n", offset, ALGEBRAIC[(offset * num_of_algebraic) + mLss],  STATES[(offset * num_of_states) + mL], ALGEBRAIC[(offset * num_of_algebraic) + tmL]); 
// }

}

__device__ void solveAnalytical(double dt, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC)
{ 

  int num_of_constants = 146;
  int num_of_states = 41;
  int num_of_algebraic = 199;
  int num_of_rates = 41;
  // int offset = threadIdx.x;
  int offset =blockIdx.x * blockDim.x + threadIdx.x; 
  // printf("current solveAnalytical offset: %d\n", offset);

  ////==============
  ////Exact solution
  ////==============
  ////INa
  STATES[(offset * num_of_states) + m] = ALGEBRAIC[(offset * num_of_algebraic) + mss] - (ALGEBRAIC[(offset * num_of_algebraic) + mss] - STATES[(offset * num_of_states) + m]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tm]);
  STATES[(offset * num_of_states) + hf] = ALGEBRAIC[(offset * num_of_algebraic) + hss] - (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hf]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + thf]);
  STATES[(offset * num_of_states) + hs] = ALGEBRAIC[(offset * num_of_algebraic) + hss] - (ALGEBRAIC[(offset * num_of_algebraic) + hss] - STATES[(offset * num_of_states) + hs]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + ths]);
  STATES[(offset * num_of_states) + j] = ALGEBRAIC[(offset * num_of_algebraic) + jss] - (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + j]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tj]);
  STATES[(offset * num_of_states) + hsp] = ALGEBRAIC[(offset * num_of_algebraic) + hssp] - (ALGEBRAIC[(offset * num_of_algebraic) + hssp] - STATES[(offset * num_of_states) + hsp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + thsp]);
  STATES[(offset * num_of_states) + jp] = ALGEBRAIC[(offset * num_of_algebraic) + jss] - (ALGEBRAIC[(offset * num_of_algebraic) + jss] - STATES[(offset * num_of_states) + jp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tjp]);
  STATES[(offset * num_of_states) + mL] = ALGEBRAIC[(offset * num_of_algebraic) + mLss] - (ALGEBRAIC[(offset * num_of_algebraic) + mLss] - STATES[(offset * num_of_states) + mL]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tmL]);
  STATES[(offset * num_of_states) + hL] = ALGEBRAIC[(offset * num_of_algebraic) + hLss] - (ALGEBRAIC[(offset * num_of_algebraic) + hLss] - STATES[(offset * num_of_states) + hL]) * exp(-dt / CONSTANTS[(offset * num_of_constants) + thL]);
  STATES[(offset * num_of_states) + hLp] = ALGEBRAIC[(offset * num_of_algebraic) + hLssp] - (ALGEBRAIC[(offset * num_of_algebraic) + hLssp] - STATES[(offset * num_of_states) + hLp]) * exp(-dt / CONSTANTS[(offset * num_of_constants) + thLp]);
  ////Ito
  STATES[(offset * num_of_states) + a] = ALGEBRAIC[(offset * num_of_algebraic) + ass] - (ALGEBRAIC[(offset * num_of_algebraic) + ass] - STATES[(offset * num_of_states) + a]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + ta]);
  STATES[(offset * num_of_states) + iF] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iF]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiF]);
  STATES[(offset * num_of_states) + iS] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iS]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiS]);
  STATES[(offset * num_of_states) + ap] = ALGEBRAIC[(offset * num_of_algebraic) + assp] - (ALGEBRAIC[(offset * num_of_algebraic) + assp] - STATES[(offset * num_of_states) + ap]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + ta]);
  STATES[(offset * num_of_states) + iFp] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iFp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiFp]);
  STATES[(offset * num_of_states) + iSp] = ALGEBRAIC[(offset * num_of_algebraic) + iss] - (ALGEBRAIC[(offset * num_of_algebraic) + iss] - STATES[(offset * num_of_states) + iSp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tiSp]);
  ////ICaL
  STATES[(offset * num_of_states) + d] = ALGEBRAIC[(offset * num_of_algebraic) + dss] - (ALGEBRAIC[(offset * num_of_algebraic) + dss] - STATES[(offset * num_of_states) + d]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + td]);
  STATES[(offset * num_of_states) + ff] = ALGEBRAIC[(offset * num_of_algebraic) + fss] - (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ff]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tff]);
  STATES[(offset * num_of_states) + fs] = ALGEBRAIC[(offset * num_of_algebraic) + fss] - (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + fs]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tfs]);
  STATES[(offset * num_of_states) + fcaf] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcaf]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tfcaf]);
  STATES[(offset * num_of_states) + fcas] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcas]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tfcas]);
  STATES[(offset * num_of_states) + jca] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + jca]) * exp(- dt / CONSTANTS[(offset * num_of_constants) + tjca]);
  STATES[(offset * num_of_states) + ffp] = ALGEBRAIC[(offset * num_of_algebraic) + fss] - (ALGEBRAIC[(offset * num_of_algebraic) + fss] - STATES[(offset * num_of_states) + ffp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tffp]);
  STATES[(offset * num_of_states) + fcafp] = ALGEBRAIC[(offset * num_of_algebraic) + fcass] - (ALGEBRAIC[(offset * num_of_algebraic) + fcass] - STATES[(offset * num_of_states) + fcafp]) * exp(-d / ALGEBRAIC[(offset * num_of_algebraic) + tfcafp]);
  STATES[(offset * num_of_states) + nca] = ALGEBRAIC[(offset * num_of_algebraic) + anca] * CONSTANTS[(offset * num_of_constants) + k2n] / ALGEBRAIC[(offset * num_of_algebraic) + km2n] - (ALGEBRAIC[(offset * num_of_algebraic) + anca] * CONSTANTS[(offset * num_of_constants) + k2n] / ALGEBRAIC[(offset * num_of_algebraic) + km2n] - STATES[(offset * num_of_states) + nca]) * exp(-ALGEBRAIC[(offset * num_of_algebraic) + km2n] * dt);
  ////IKr
  STATES[(offset * num_of_states) + xrf] = ALGEBRAIC[(offset * num_of_algebraic) + xrss] - (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrf]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txrf]);
  STATES[(offset * num_of_states) + xrs] = ALGEBRAIC[(offset * num_of_algebraic) + xrss] - (ALGEBRAIC[(offset * num_of_algebraic) + xrss] - STATES[(offset * num_of_states) + xrs]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txrs]);
  ////IKs
  STATES[(offset * num_of_states) + xs1] = ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] - (ALGEBRAIC[(offset * num_of_algebraic) + xs1ss] - STATES[(offset * num_of_states) + xs1]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txs1]);
  STATES[(offset * num_of_states) + xs2] = ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] - (ALGEBRAIC[(offset * num_of_algebraic) + xs2ss] - STATES[(offset * num_of_states) + xs2]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txs2]);
  ////IK1
  STATES[(offset * num_of_states) + xk1] = ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] - (ALGEBRAIC[(offset * num_of_algebraic) + xk1ss] - STATES[(offset * num_of_states) + xk1]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + txk1]);
  ////INaCa
  ////INaK
  ////IKb
  ////INab
  ////ICab
  ///IpCa
  ////Diffusion fluxes
  ////RyR receptors
  STATES[(offset * num_of_states) + Jrelnp] = ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] - (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_inf] - STATES[(offset * num_of_states) + Jrelnp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tau_rel]);
  STATES[(offset * num_of_states) + Jrelp] = ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] - (ALGEBRAIC[(offset * num_of_algebraic) + Jrel_infp] - STATES[(offset * num_of_states) + Jrelp]) * exp(-dt / ALGEBRAIC[(offset * num_of_algebraic) + tau_relp]);
  ////SERCA Pump
  ////Calcium translocation
  //
  ////=============================
  ////Approximated solution (Euler)
  ////=============================
  ////ICaL
  //STATES[jca] = STATES[jca] + RATES[jca] * dt;
  ////CaMK
  STATES[(offset * num_of_states) + CaMKt] = STATES[(offset * num_of_states) + CaMKt] + RATES[(offset * num_of_rates) + CaMKt] * dt;
  ////Membrane potential
  STATES[(offset * num_of_states) + V] = STATES[(offset * num_of_states) + V] + RATES[(offset * num_of_rates) + V] * dt;
  ////Ion Concentrations and Buffers
  STATES[(offset * num_of_states) + nai] = STATES[(offset * num_of_states) + nai] + RATES[(offset * num_of_rates) + nai] * dt;
  STATES[(offset * num_of_states) + nass] = STATES[(offset * num_of_states) + nass] + RATES[(offset * num_of_rates) + nass] * dt;
  STATES[(offset * num_of_states) + ki] = STATES[(offset * num_of_states) + ki] + RATES[(offset * num_of_rates) + ki] * dt;
  STATES[(offset * num_of_states) + kss] = STATES[(offset * num_of_states) + kss] + RATES[(offset * num_of_rates) + kss] * dt;
  STATES[(offset * num_of_states) + cai] = STATES[(offset * num_of_states) + cai] + RATES[(offset * num_of_rates) + cai] * dt;
  STATES[(offset * num_of_states) + cass] = STATES[(offset * num_of_states) + cass] + RATES[(offset * num_of_rates) + cass] * dt;
  STATES[(offset * num_of_states) + cansr] = STATES[(offset * num_of_states) + cansr] + RATES[(offset * num_of_rates) + cansr] * dt;
  STATES[(offset * num_of_states) + cajsr] = STATES[(offset * num_of_states) + cajsr] + RATES[(offset * num_of_rates) + cajsr] * dt; 
  //========================
  //Full Euler Approximation
  //========================
  //STATES[V] = STATES[V] + RATES[V] * dt;
  //STATES[CaMKt] = STATES[CaMKt] + RATES[CaMKt] * dt;
  //STATES[cass] = STATES[cass] + RATES[cass] * dt;
  //STATES[nai] = STATES[nai] + RATES[nai] * dt;
  //STATES[nass] = STATES[nass] + RATES[nass] * dt;
  //STATES[ki] = STATES[ki] + RATES[ki] * dt;
  //STATES[kss] = STATES[kss] + RATES[kss] * dt;
  //STATES[cansr] = STATES[cansr] + RATES[cansr] * dt;
  //STATES[cajsr] = STATES[cajsr] + RATES[cajsr] * dt;
  //STATES[cai] = STATES[cai] + RATES[cai] * dt;
  //STATES[m] = STATES[m] + RATES[m] * dt;
  //STATES[hf] = STATES[hf] + RATES[hf] * dt;
  //STATES[hs] = STATES[hs] + RATES[hs] * dt;
  //STATES[j] = STATES[j] + RATES[j] * dt;
  //STATES[hsp] = STATES[hsp] + RATES[hsp] * dt;
  //STATES[jp] = STATES[jp] + RATES[jp] * dt;
  //STATES[mL] = STATES[mL] + RATES[mL] * dt;
  //STATES[hL] = STATES[hL] + RATES[hL] * dt;
  //STATES[hLp] = STATES[hLp] + RATES[hLp] * dt;
  //STATES[a] = STATES[a] + RATES[a] * dt;
  //STATES[iF] = STATES[iF] + RATES[iF] * dt;
  //STATES[iS] = STATES[iS] + RATES[iS] * dt;
  //STATES[ap] = STATES[ap] + RATES[ap] * dt;
  //STATES[iFp] = STATES[iFp] + RATES[iFp] * dt;
  //STATES[iSp] = STATES[iSp] + RATES[iSp] * dt;
  //STATES[d] = STATES[d] + RATES[d] * dt;
  //STATES[ff] = STATES[ff] + RATES[ff] * dt;
  //STATES[fs] = STATES[fs] + RATES[fs] * dt;
  //STATES[fcaf] = STATES[fcaf] + RATES[fcaf] * dt;
  //STATES[fcas] = STATES[fcas] + RATES[fcas] * dt;
  //STATES[jca] = STATES[jca] + RATES[jca] * dt;
  //STATES[ffp] = STATES[ffp] + RATES[ffp] * dt;
  //STATES[fcafp] = STATES[fcafp] + RATES[fcafp] * dt;
  //STATES[nca] = STATES[nca] + RATES[nca] * dt;
  //STATES[xrf] = STATES[xrf] + RATES[xrf] * dt;
  //STATES[xrs] = STATES[xrs] + RATES[xrs] * dt;
  //STATES[xs1] = STATES[xs1] + RATES[xs1] * dt;
  //STATES[xs2] = STATES[xs2] + RATES[xs2] * dt;
  //STATES[xk1] = STATES[xk1] + RATES[xk1] * dt;
  //STATES[Jrelnp] = STATES[Jrelnp] + RATES[Jrelnp] * dt;
  //STATES[Jrelp] = STATES[Jrelp] + RATES[Jrelp] * dt;
}

// ------------------------------------------
__device__ void applyDrugEffect(double conc, double *ic50, double epsilon, double *CONSTANTS)
{
  // int offset = threadIdx.x;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int num_of_constants = 146;
// cek lagi value ini, apakah dia bervariasi sesuai dengan samplenya? 
CONSTANTS[GK1+(offset * num_of_constants)] = CONSTANTS[GK1+(offset * num_of_constants)] * ((ic50[2 + (offset*14)] > 10E-14 && ic50[3+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[2+ (offset*14)],ic50[3+ (offset*14)])) : 1.);
CONSTANTS[GKr+(offset * num_of_constants)] = CONSTANTS[GKr+(offset * num_of_constants)] * ((ic50[12+ (offset*14)] > 10E-14 && ic50[13+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[12+ (offset*14)],ic50[13+ (offset*14)])) : 1.);
CONSTANTS[GKs+(offset * num_of_constants)] = CONSTANTS[GKs+(offset * num_of_constants)] * ((ic50[4 + (offset*14)] > 10E-14 && ic50[5+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[4+ (offset*14)],ic50[5+ (offset*14)])) : 1.);
CONSTANTS[GNaL+(offset * num_of_constants)] = CONSTANTS[GNaL+(offset * num_of_constants)] = CONSTANTS[GNaL+(offset * num_of_constants)] * ((ic50[8+ (offset*14)] > 10E-14 && ic50[9+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[8+ (offset*14)],ic50[9+ (offset*14)])) : 1.);
CONSTANTS[GNa+(offset * num_of_constants)] = CONSTANTS[GNa+(offset * num_of_constants)] * ((ic50[6 + (offset*14)] > 10E-14 && ic50[7+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[6+ (offset*14)],ic50[7+ (offset*14)])) : 1.);
CONSTANTS[Gto+(offset * num_of_constants)] = CONSTANTS[Gto+(offset * num_of_constants)] * ((ic50[10 + (offset*14)] > 10E-14 && ic50[11+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[10+ (offset*14)],ic50[11+ (offset*14)])) : 1.);
CONSTANTS[PCa+(offset * num_of_constants)] = CONSTANTS[PCa+(offset * num_of_constants)] * ( (ic50[0 + (offset*14)] > 10E-14 && ic50[1+ (offset*14)] > 10E-14) ? 1./(1.+pow(conc/ic50[0+ (offset*14)],ic50[1+ (offset*14)])) : 1.);
}


char buffer[255];
double ic50[14*2000];
// __shared__ drug_t *d_ic50;
double *d_concs;

int get_IC50_data_from_file(const char* file_name, double *ic50)
{
    /*
    a host function to take all samples from the file, assuming each sample has 14 features.

    it takes the file name, and an ic50 (already declared in 1D, everything become 1D)
    as a note, the data will be stored in 1D array, means this functions applies flatten.

    it returns 'how many samples were detected?' in integer.
    */
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

__global__ void do_drug_sim_analytical(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *states,
                                       double *ical, double *inal){
    unsigned short sample_id, input_counter;
    input_counter = 0;
    sample_id = blockIdx.x * blockDim.x + threadIdx.x;
  
    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    //int num_of_rates = 41;

    double tcurr = 0.000001, dt = 0.005, tmax;
    double max_time_step = 1.0, time_point = 25.0;
    double dt_set;

    // files for storing results
    // time-series result
    // FILE *fp_vm, *fp_inet, *fp_gate;

    // features
    // double inet, qnet;

    // looping counter
    // unsigned short idx;
  
    // simulation parameters
    // double dtw = 2.0;
    // const char *drug_name = "bepridil";
    const double bcl = 2000; // bcl is basic cycle length
    
    // const double inet_vm_threshold = -88.0;
    const unsigned short pace_max = 1000;
    // const unsigned short pace_max = 6;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    // const unsigned short last_drug_check_pace = 250;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    double conc = 99.0;
    // double conc = 0.0;


    // printf("Core %d:\n",sample_id);
    initConsts(d_CONSTANTS, d_STATES);

    applyDrugEffect(conc,d_ic50,1E-14,d_CONSTANTS);

    d_CONSTANTS[stim_period + (sample_id * num_of_constants)] = bcl;

    // generate file for time-series output

    tmax = pace_max * bcl;
    double time_record_limit = tmax - bcl;
    int pace_count = 0;
  
    // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt_set[sample_id], tcurr, d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)]);

    while (tcurr<tmax){
        dt_set = set_time_step(tcurr, time_point, max_time_step, d_CONSTANTS, d_RATES); // cara nge cek nya gimana ya???
        computeRates(tcurr, d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC); // ini nih yang bikin nan
        // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt_set[sample_id], tcurr, d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)]);
        //Compute the correct/accepted time step
        //printf("core %d pas update dt nilai bandingan if nya: %lf dibanding %lf dan verdict %d\n", sample_id, floor((tcurr + dt_set) / bcl), floor(tcurr / bcl), (floor((tcurr + dt_set) / bcl) == floor(tcurr / bcl))  );
        if (floor((tcurr + dt_set) / bcl) == floor(tcurr / bcl)) { // mungkin kena di sini, karena pas di debugger, tcurr itu nol, jadi ada div by zero
        // tammbahan: setelah di cek ke codingan CPU, kedua belah if harusnya sama secara angka, gak cuman true terus di verdict
        // sementara di codingan GPU, walaupun kanan nol terus, kiri satu terus, tetep di bilang true
          // printf("timestep not corrected in core %d \n", sample_id);
          dt = dt_set;
        }
        else {
          dt = (floor(tcurr / bcl) + 1) * bcl - tcurr;
          pace_count++;
          // printf("core %d, pace_count: %d, tcurr: %lf\n", sample_id, pace_count, tcurr);
          // printf("timestep corrected in core %d \n", sample_id);
        }

        solveAnalytical(dt, d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC);
        tcurr = tcurr + dt;
        if (tcurr >= time_record_limit){
        //printf("%d,%lf,%lf,%lf,%lf,%lf\n", sample_id, dt_set, tcurr, d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)], d_CONSTANTS[GKs+(sample_id * num_of_constants)]);
        time[input_counter + sample_id] = tcurr;
        states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
        ical[input_counter + sample_id] = d_ALGEBRAIC[(sample_id * num_of_algebraic) + ICaL];
        inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)];
        input_counter++;
        }
    }
    // printf("\n");
    // for (int z=0+(sample_id*146);z<(sample_id*146)+146;z++){
    //       printf("Core %d CONSTANTS[%d]: %lf \n",sample_id, z, d_CONSTANTS[z]);
    // }

    


    // for (int z=0+(sample_id*146);z<(sample_id*146)+146;z++){
        // printf("Core %d CONSTANTS[%d]: %lf \n",sample_id, z, d_CONSTANTS[z]);
    // }
    //printf("\n");
    // for (int z=0+(sample_id*41);z<(sample_id*41)+41;z++){
        // printf("Core %d STATES[%d]: %lf \n",sample_id, z, d_STATES[z]);
    // }

 
}

int main()
{
    double *d_ic50;
    double *d_ALGEBRAIC;
    double *d_CONSTANTS;
    double *d_RATES;
    double *d_STATES;

    double *time;
    double *states;
    double *ical;
    double *inal;

    // input variables for cell simulation

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    int num_of_rates = 41;

    snprintf(buffer, sizeof(buffer),
      // "./drugs/chlorpromazine/IC50_samples100.csv"
      "./IC50_samples100.csv"
      );
    int sample_size = get_IC50_data_from_file(buffer, ic50);
    if(sample_size == 0)
        printf("Something problem with the IC50 file!\n");
    else if(sample_size > 2000)
        printf("Too much input! Maximum sample data is 2000!\n");
    printf("Sample size: %d\n",sample_size);
   
    // double ALGEBRAIC[num_of_algebraic * sample_size];
    // double CONSTANTS[num_of_constants * sample_size];
    // double RATES[num_of_rates * sample_size]; 
    // double STATES[num_of_states * sample_size];
    printf("preparing GPU memory space \n");
    cudaMalloc(&d_ALGEBRAIC, num_of_algebraic * sample_size * sizeof(double));
    cudaMalloc(&d_CONSTANTS, num_of_constants * sample_size * sizeof(double));
    cudaMalloc(&d_RATES, num_of_rates * sample_size * sizeof(double));
    cudaMalloc(&d_STATES, num_of_states * sample_size * sizeof(double));
    // prep for 1 cycle plus a bit (700 * sample_size)
    cudaMalloc(&time, num_of_states * sample_size * 2100 * sizeof(double)); 
    cudaMalloc(&states, num_of_states * sample_size * 2100 * sizeof(double));
    cudaMalloc(&ical, num_of_states * sample_size * 2100 * sizeof(double));
    cudaMalloc(&inal, num_of_states * sample_size * 2100 * sizeof(double));
    

    // printf("cuda malloc status for algebraic: %d\n", cudaMalloc(&d_ALGEBRAIC, num_of_algebraic * sample_size * sizeof(double))) ;
    // printf("cuda malloc status for constants: %d\n", cudaMalloc(&d_CONSTANTS, num_of_constants * sample_size * sizeof(double)));
    // printf("cuda malloc status for rates: %d\n", cudaMalloc(&d_RATES, num_of_rates * sample_size * sizeof(double)));
    // printf("cuda malloc status for states: %d\n",cudaMalloc(&d_STATES, num_of_states * sample_size * sizeof(double)));
    printf("Copying sample files to GPU memory space \n");
    cudaMalloc(&d_ic50, sample_size * 14 * sizeof(double));
    cudaMemcpy(d_ic50, ic50, sample_size * 14 * sizeof(double), cudaMemcpyHostToDevice);


    tic();

    // double *dt_set;
    // cudaMalloc(&dt_set, sample_size * sizeof(double) );
    // printf("algebraic: %zu, constants: %zu, rates: %zu, states: %zu, dt_set: %zu\n",sizeof(d_ALGEBRAIC), sizeof(d_CONSTANTS), sizeof(d_RATES), sizeof(d_STATES), sizeof(dt_set));
    // printf("samples detected: %d\n",sample_size);
    // printf("core,dt_set,tcurr,states,rates,GKs\n");
    // printf("%d, %d",sample_size/100,sample_size%100);
    printf("doing simulation.... \n");
    do_drug_sim_analytical<<<sample_size/50,50>>>(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, time, states, ical, inal);
                                      //block, core
    cudaDeviceSynchronize();
    
    printf("copying the data back to the CPU \n");

    ////// copy the data back to CPU, and write them into file ////////
    double h_states[2100 * sample_size];
    double h_time[2100 * sample_size];
    double h_ical[2100 * sample_size];
    double h_inal[2100 * sample_size];

    cudaMemcpy(h_states, states, sample_size * 2100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time, time, sample_size * 2100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ical, ical, sample_size * 2100 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inal, inal, sample_size * 2100 * sizeof(double), cudaMemcpyDeviceToHost);

    FILE *writer;
    // sample loop
    for (int sample_id = 0; sample_id<sample_size; sample_id++){
      // double result_buffer[2][7000];
      // result_buffer[0][...] is for the time, and result_buffer[1][...] is for the states,
      char sample_str[ENOUGH];
      char filename[150] = "./result/bepridil_mods/10state_sample";
      sprintf(sample_str, "%d", sample_id);
      strcat(filename,sample_str);
      strcat(filename,".csv");

      writer = fopen(filename,"w");
      fprintf(writer, "time,state,ICaL,INaL\n"); 
      for (int datapoint = 0; datapoint<2100; datapoint++){
        if (h_time[ sample_id + (datapoint * sample_size)] == 0.0) {continue;}
        fprintf(writer, "%lf,%lf,%lf,%lf\n",
        h_time[ sample_id + (datapoint * sample_size)], 
        h_states[ sample_id + (datapoint * sample_size)], 
        h_ical[ sample_id + (datapoint * sample_size)], 
        h_inal[ sample_id + (datapoint * sample_size)]
        );
      }
      fclose(writer);
    }
    
    // char buffer[255];

    // snprintf(buffer, sizeof(buffer), "states_trial.plt");

    // writer = fopen(buffer, "w");

    // fprintf(writer, "core,dt_set,tcurr,states\n");
    // unsigned short sample_id;

    // for( sample_id = 0;
    //     sample_id < sample_size;  // ---> sample loop
    //     sample_id ++ )
    // { // begin sample loop
    //     printf("Sample_ID:%d \nData: ",
    //     sample_id );
        
    //     for (int z=0+(sample_id*14);z<(sample_id*14)+14;z++){
    //         fprintf(writer, "%d,%lf,%lf,%lf\n", sample_id, dt_set[sample_id], tcurr, d_STATES[V + (sample_id * num_of_states)]);
    //     }
    //     printf("\n");

    // } // end sample loop
    toc();
    
    return 0;
}
