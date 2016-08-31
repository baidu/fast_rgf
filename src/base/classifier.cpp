/************************************************************************
 *  classifier.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "classifier.h"

int TrainLoss::str2loss(string loss_str)
{
  int loss= TrainLoss::INVALID;
  if (loss_str.compare("MODLS") ==0) loss= TrainLoss::MODLS;
  if (loss_str.compare("LOGISTIC") ==0) loss= TrainLoss::LOGISTIC;
  if (loss_str.compare("LS") ==0) loss= TrainLoss::LS;
  if (loss==TrainLoss::INVALID) {
    cerr << "loss " << loss_str << " is invalid" <<endl;
    cerr << "valid values are " << "MODLS or LS or LOGISTIC" <<endl;
    exit(-1);
  }
  return loss;
}

string TrainLoss::loss2str(int loss)
{
  switch(loss) {
  case LS:
    return "least squares loss";
  case MODLS:
    return "modified least squares loss";
  case LOGISTIC:
    return "logistic loss";
  default:
    return "invalid loss";
  }
}

double TrainLoss::binary_loss(int loss, double scr, double y)
{
  double tmp;
  switch(loss) {
  case LS:
    tmp=scr-y;
    return tmp*tmp;
  case MODLS:
    tmp=scr*y;
    return (tmp>1)? 0: (tmp-1)*(tmp-1);
  case LOGISTIC:
    tmp=scr*y;
    return log(1+exp(-tmp));
  default:
    cerr << "invalid loss" <<endl;
    exit(-1);
  }
}

void BinaryTestStat::update(double y, double scr, bool pred)
{
  bool truth=false;

  if (_y_type.type== Target::BINARY) {
    truth=_y_type.binary_label(y);
    total_loss += TrainLoss::binary_loss(_loss,scr,truth?1.0:-1.0);
  }
  else {
    total_loss += TrainLoss::binary_loss(_loss,scr,y);
  }
  num++;
  if (truth) {
    if (pred) tp++;
    else fn++;
  }
  else {
    if (pred) fp++;
    else tn++;
  }
  if (keep_results) {
    _results.push_back(TestResult(scr,y));
  }
}


void BinaryTestStat::print(ostream & os)
{

  if (_y_type.type==Target::BINARY) {
    double a=auc();
    double value= total_loss/num;
    os << TrainLoss::loss2str(_loss) << "=" << value << " ";
    os << 
      "tp=" << tp << " fp=" << fp << " tn="<< tn << " fn=" << fn <<endl;
    os << "precision="<<precision() << " recall=" << recall()
       << " Fb1=" << fb1() 
       << " accuracy=" << accuracy();
    if (a<=0) os << endl;
    else os << " auc=" << a <<endl;
  }
  else {
    double value= total_loss/num;
    os << TrainLoss::loss2str(_loss) << "=" << value <<endl;
  }
}

void BinaryTestStat::clear()
{
  tp=fp=tn=fn=0;
  _results.clear();
}

void BinaryTestStat::roc(size_t tp0, size_t tn0, double & tpr, double & fpr)
{
  size_t pp=tp+fn;
  size_t nn=fp+tn;
  tpr= tp0/(pp+1e-10);
  fpr= (nn-tn0)/(nn+1e-10);
}

double BinaryTestStat::auc()
{
  if (!keep_results) return 0.0;
  
  
  vector<size_t> tp_vec;
  
  vector<size_t> tn_vec;

  {
    sort(_results.begin(),_results.end());
    tp_vec.clear();
    tn_vec.clear();

    
    size_t my_tp=tp+fn;
    size_t my_tn=0;
    tp_vec.push_back(my_tp);
    tn_vec.push_back(my_tn);

    for (size_t i=0; i<_results.size(); i++) {
      bool truth= _y_type.binary_label(_results[i].y);
      if (truth) my_tp--;
      else my_tn++;
      if (i<_results.size()-1 && _results[i].scr==_results[i+1].scr) continue;
      tp_vec.push_back(my_tp);
      tn_vec.push_back(my_tn);
    }
  }

  double tpr1, tpr2;
  double fpr1, fpr2;
  double a=0;

  roc(tp_vec[0],tn_vec[0], tpr1,fpr1);
  for (size_t i=1; i< tp_vec.size(); i++) {
    roc(tp_vec[i],tn_vec[i], tpr2,fpr2);
    
    a += 0.5*(tpr1+tpr2)*(fpr1-fpr2);

    tpr1=tpr2;
    fpr1=fpr2;
  }
  return a;
}
