/************************************************************************
 *  training_target.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _DECISIONTREE_TRAINER_TRAINING_TARGET_H

#define _DECISIONTREE_TRAINER_TRAINING_TARGET_H

#include "dtree.h"




namespace _decisionTreeTrainer
{
  
  class YW_struct {
  public:
    
    double y;
    
    float w;
    
    float _empty;
    
    inline void set(double yp, double wp) {
      y=yp;
      w=wp;
    }

    inline void add(double yp) {
      y+=yp;
      w+=1.0;
    }
    inline void add(double yp, double wp) {
      y+=yp;
      w+=wp;
    }
    inline void add(const YW_struct & p) {
      y+=p.y;
      w+=p.w;
    }
  };
  
  
  class YW0_struct {
  public:
    
    float y;
    
    float w;

    inline void set(double yp, double wp) {
      y=yp;
      w=wp;
    }
  };

 
  template<typename T>
  inline void swap_arr(T * arr, pair<train_size_t, train_size_t> *swap_pairs, train_size_t swap_size)
  {
    for (train_size_t i=0; i< swap_size; i++) {
      pair<train_size_t,train_size_t> ii=swap_pairs[i];
      auto tmp=arr[ii.first];
      arr[ii.first]=arr[ii.second];
      arr[ii.second]=tmp;
    }
  }


  
  class TrainTarget {
    
    YW0_struct *_yw=nullptr;
    
    double _prediction0=0;
  public:
    
    const int vec_width=8;
    
    int loss;

    
    double * residues=nullptr;
    
    float * weights=nullptr;
    
    bool * labels=nullptr;

    
    train_size_t * index=nullptr;
    
    
    YW0_struct * yw_data_arr() {
      return _yw;
    }

    
    double prediction0() {
      return _prediction0;
    }

    inline void yw_LS_set(YW0_struct & yw, double w, double res)
    {
      double nfp=-(res);
      const double fpp=1.0;
      yw.set(nfp*w,fpp*w);
    }
    inline void yw_LS_add(YW_struct & yw, double w, double res)
    {
      double nfp=-(res);
      const double fpp=1.0;
      yw.add(nfp*w,fpp*w);
    }

    inline void yw_ModLS_set(YW0_struct & yw, double w, double s, double res)
    {
      double fpp= (res*s>1.0)?max(0.1,1.0-5*(res*s-1)):1.0;
      double nfp= (res*s>1.0)? 0: (s-res);
      yw.set(nfp*w,fpp*w);
    }
    inline void yw_ModLS_add(YW_struct & yw, double w, double s, double res)
    {
      double fpp= (res*s>1.0)?max(0.1,1.0-5*(res*s-1)):1.0;
      double nfp= (res*s>1.0)? 0: (s-res);
      yw.add(nfp*w,fpp*w);
    }

    inline void yw_Logistic_set(YW0_struct & yw, double w, double is_inclass, double res)
    {
      double p= 1.0/(1.0+exp(-res));
      double nfp= is_inclass-p;
      double fpp= p*(1-p);
      yw.set(nfp*w,fpp*w);
    }
    inline void yw_Logistic_add(YW_struct & yw, double w, double is_inclass, double res)
    {
      double p= 1.0/(1.0+exp(-res));
      double nfp= is_inclass-p;
      double fpp= p*(1-p);
      yw.add(nfp*w,fpp*w);
    }
    inline void yw_Logistic_set_vect(double pred0, size_t pos)
    {
      int j;
      float ww[vect_width],yy[vect_width],qq[vect_width];
      if (weights) {
	for (j=0; j<vect_width; j++) {
	  ww[j]=weights[pos+j];
	}
      }
      else {
	for (j=0; j<vect_width; j++) {
	  ww[j]=1.0;
	}
      }
      for (j=0; j<vect_width; j++) {
	yy[j]=labels[pos+j]?1.0:0.0;
	qq[j]=-(residues[pos+j]+pred0);
      }
      for (j=0; j<vect_width; j++) {
	qq[j]=exp(qq[j]);
      }
      float p[vect_width],nfp[vect_width],fpp[vect_width];
      for (j=0; j<vect_width; j++) {
	p[j]=1.0/(1.0+qq[j]);
      }
      for (j=0; j<vect_width; j++) {
	nfp[j]= (yy[j]-p[j])*ww[j];
      }
      for (j=0; j<vect_width; j++) {
	fpp[j]= p[j]*(1-p[j])*ww[j];
      }
      for (j=0; j<vect_width; j++) {
	_yw[pos+j].set(nfp[j],fpp[j]);
      }
    }

    inline void yw_Logistic_add_vect(YW_struct *yw_arr, float * weights, bool *labels, double * residues, int *my_i, size_t pos)
    {
      int j;
      float ww[vect_width],yy[vect_width],qq[vect_width];
      if (weights) {
	for (j=0; j<vect_width; j++) {
	  ww[j]=weights[pos+j];
	}
      }
      else {
	for (j=0; j<vect_width; j++) {
	  ww[j]=1.0;
	}
      }
      for (j=0; j<vect_width; j++) {
	yy[j]=labels[pos+j]?1.0:0.0;
	qq[j]=-(residues[pos+j]);
      }
      for (j=0; j<vect_width; j++) {
	qq[j]=exp(qq[j]);
      }
      float p[vect_width],nfp[vect_width],fpp[vect_width];
      for (j=0; j<vect_width; j++) {
	p[j]=1.0/(1.0+qq[j]);
      }
      for (j=0; j<vect_width; j++) {
	nfp[j]= (yy[j]-p[j])*ww[j];
      }
      for (j=0; j<vect_width; j++) {
	fpp[j]= p[j]*(1-p[j])*ww[j];
      }
      for (j=0; j<vect_width; j++) {
	yw_arr[my_i[pos+j]].add(nfp[j],fpp[j]);
      }
    }


    
    
    void compute_yw(train_size_t size, double pred0, int nthreads) {
#ifdef USE_OMP
      omp_set_num_threads(nthreads);
#endif
      train_size_t i;
      if (loss==TrainLoss::LS) {
#pragma omp parallel for
	for (i=0; i<size; i++) {
	  yw_LS_set(_yw[i],weights?weights[i]:1.0,residues[i]+pred0);
	}
      }
      if (loss==TrainLoss::MODLS) {
#pragma omp parallel for	
	for (i=0; i<size; i++) {
	  yw_ModLS_set(_yw[i],weights?weights[i]:1.0,labels[i]?1:-1,residues[i]+pred0);
	}
      }
      if (loss==TrainLoss::LOGISTIC) {
	int k=(vect_width==0)?0:(size/vect_width);
#pragma omp parallel for
	for (i=0; i<k; i++) {
	  yw_Logistic_set_vect(pred0,i*vect_width);
	}
	for (i=k*vect_width; i<size; i++) {
	  yw_Logistic_set(_yw[i],weights?weights[i]:1.0,labels[i]?1.0:0.0,residues[i]+pred0);
	}
      }
      _prediction0=pred0;
    }
    
    void compute_yw(int *reverse_index, train_size_t b, train_size_t e, YW_struct * yw, int num_yw) {
      train_size_t i,j;
      unique_ptr<YW_struct []> my_yw(new YW_struct [num_yw]);
      auto my_w= weights?(&weights[b]):nullptr;
      auto my_r= &residues[b];
      auto my_i= &reverse_index[b];
      auto my_L= labels?(&labels[b]):nullptr;
      auto size= e-b;

      for (j=0; j<num_yw; j++) {
	my_yw[j].set(0,0);
      }
      if (loss==TrainLoss::LS) {
	for (i=0; i<size; i++) {
	  j= my_i[i];
	  yw_LS_add(my_yw[j],my_w?my_w[i]:1.0,my_r[i]);
	}
      }
      if (loss==TrainLoss::MODLS) {
	for (i=0; i<size; i++) {
	  j= my_i[i];
	  yw_ModLS_add(my_yw[j],my_w?my_w[i]:1.0,my_L[i]?1.0:-1.0,my_r[i]);
	}
      }
      if (loss==TrainLoss::LOGISTIC) {
	int k=(vect_width==0)?0:(size/vect_width);
	for (i=0; i<k; i++) {
	  yw_Logistic_add_vect(my_yw.get(),my_w,my_L,my_r,my_i,i*vect_width);
	}
	for (i=k*vect_width; i<size; i++) {
	  j= my_i[i];
	  yw_Logistic_add(my_yw[j],my_w?my_w[i]:1.0,my_L[i]?1.0:0.0,my_r[i]);
	}
      }
      for (j=0; j<num_yw; j++) yw[j]=my_yw[j];
    }
    
    
    void set(size_t nrows, double *y, double *scr, float * w, string loss_str, Target & y_type) {
      loss= TrainLoss::str2loss(loss_str);
      
      if (_yw==nullptr) _yw = new YW0_struct [nrows];
      
      if (w !=nullptr) {
	if (weights==nullptr) weights= new float [nrows];
	memcpy(weights,w,sizeof(*weights)*nrows);
      }

      if (residues==nullptr) {
	residues=new double [nrows];
      }

      if (index==nullptr) {
	index= new train_size_t [nrows];
      }
      for (train_size_t i=0; i<nrows; i++) index[i]=i;

      if (loss==TrainLoss::LS) { 
	if (y_type.type==Target::REAL) {
	  for (size_t i=0; i<nrows; i++) {
	    residues[i]=((scr==nullptr)?0:scr[i])-y[i];
	  }
	}
	else { 
	  for (size_t i=0; i<nrows; i++) {
	    double yy=y_type.binary_label(y[i])?1.0:-1.0;
	    residues[i]=((scr==nullptr)?0:(scr[i])-yy);
	  }
	}
      }
      else { 
	if (y_type.type == Target::REAL) {
	  cerr << endl <<
	    "error in decision tree training: real valued target cannot use binary classification loss "
	       <<endl;
	  exit(-1);
	}
	if (labels==nullptr) {
	  labels=new bool [nrows];
	}
	for (size_t i=0; i<nrows; i++) {
	  residues[i]=((scr==nullptr)?0:scr[i]);
	  labels[i]=y_type.binary_label(y[i]);
	}
      }
      
    }

    
    void copy_back(size_t nrows, double *y, double *scr) {
      if (scr==nullptr) return;
      
      if (loss==TrainLoss::LS) {
	for (size_t i=0; i<nrows; i++) {
	  scr[i]=residues[i]+y[i];
	}
      }
      else {
	for (size_t i=0; i<nrows; i++) {
	  scr[i]=residues[i];
	}
      }
    }

    
    void swap(pair<train_size_t,train_size_t> *swap_pairs, train_size_t swap_size)
    {
      if (residues !=nullptr) {
	swap_arr<double>(residues, swap_pairs,swap_size);
      }
      if (weights !=nullptr) {
	swap_arr<float>(weights, swap_pairs,swap_size);
      }
      if (labels !=nullptr) {
	swap_arr<bool>(labels, swap_pairs,swap_size);
      }
      if (index !=nullptr) {
	swap_arr<train_size_t>(index, swap_pairs,swap_size);
      }
    }

    
    TrainTarget shift(train_size_t pos) {
      TrainTarget result=*this;
      if (result._yw !=nullptr) result._yw +=pos;
      if (result.residues !=nullptr) result.residues+=pos;
      if (result.weights !=nullptr) result.weights +=pos;
      if (result.labels !=nullptr) result.labels +=pos;
      if (result.index !=nullptr) result.index +=pos;
      return result;
    }

    
    void clear() {
      delete [] _yw;
      delete [] residues;
      delete [] weights;
      delete [] labels;
      delete [] index;
      _yw=nullptr;
      residues=nullptr;
      weights=nullptr;
      labels=nullptr;
      index=nullptr;
    }
  };

  
  inline double solve_L1_L2(double w, double y, double lamL1, double lamL2, double &x)
  {
    w = w + lamL2+1e-10;
    x = y/w;
    double x0= lamL1/w;
    x= (x>x0)? (x-x0) : ((x<-x0)? (x+x0): 0.0);
    return 0.5 * w * x*x - x* y + lamL1 * abs(x);
  }

}




#endif
