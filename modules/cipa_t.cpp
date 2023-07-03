#include "cipa_t.hpp"

cipa_t::cipa_t()
{

}

cipa_t::cipa_t( const cipa_t &source )
{
  copy(source);
}


cipa_t& cipa_t::operator=(const cipa_t & source)
{
  if( this != &source ) copy(source);

  return *this;
}

void cipa_t::copy(const cipa_t &source)
{
  qnet = source.qnet;
  inal_auc = source.inal_auc;
  ical_auc = source.ical_auc;
  dvmdt_repol = source.dvmdt_repol;
  dvmdt_max = source.dvmdt_max;
  vm_peak = source.vm_peak;
  vm_valley = source.vm_valley;
  vm_dia = source.vm_dia;
  apd90 = source.apd90;
  apd50 = source.apd50;
  ca_peak = source.ca_peak;
  ca_valley = source.ca_valley;
  ca_dia = source.ca_dia;
  cad90 = source.cad90;
  cad50 = source.cad50;
  vm_data.clear();
  cai_data.clear();
  ires_data.clear();
  vm_data.insert( (source.vm_data).begin(), (source.vm_data).end() );
  dvmdt_data.insert( (source.dvmdt_data).begin(), (source.dvmdt_data).end() );
  cai_data.insert( (source.cai_data).begin(), (source.cai_data).end() );  
  ires_data.insert( (source.ires_data).begin(), (source.ires_data).end() );  
}

void cipa_t::init(const double vm_val, const double ca_val)
{
  qnet = 0.;
  inal_auc = 0.;
  ical_auc = 0.;
  dvmdt_repol = -999;
  dvmdt_max = -999;
  vm_peak = -999;
  vm_valley = vm_val;
  vm_dia = -999;
  apd90 = 0.;
  apd50 = 0.;
  ca_peak = -999;
  ca_valley = ca_val;
  ca_dia = -999;
  cad90 = 0.;
  cad50 = 0.;
  vm_data.clear();
  dvmdt_data.clear();
  cai_data.clear();
  ires_data.clear();
}
