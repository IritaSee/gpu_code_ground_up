#ifndef CIPA_T_HPP
#define CIPA_T_HPP

#include <map>
#include <string>

using std::multimap;
using std::string;

struct cipa_t{
  double qnet;
  double inal_auc;
  double ical_auc;
  double dvmdt_repol;
  double dvmdt_max;
  double vm_peak;
  double vm_valley;
  double vm_dia;
  double apd90;
  double apd50;
  double apd_tri;
  double ca_peak;
  double ca_valley;
  double ca_dia;
  double cad90;
  double cad50;
  double cad_tri;
  multimap<double, double> vm_data;
  multimap<double, double> dvmdt_data;
  multimap<double, double> cai_data;
  multimap<double, string> ires_data;

  cipa_t();
  cipa_t( const cipa_t &source );
  cipa_t& operator=(const cipa_t & source);
  void copy(const cipa_t &source);
  void init(const double vm_val, const double ca_val);


};


#endif
