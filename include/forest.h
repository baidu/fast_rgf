/************************************************************************
 *  forest.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _RGF_FOREST_H

#define _RGF_FOREST_H

#include "dtree.h"

namespace rgf {

  
  template<typename d_t, typename i_t, typename v_t>
  class DecisionForest: public BinaryClassifier<d_t,i_t,v_t> {

    
    vector<DecisionTree<d_t,i_t,v_t> > _dtree_vec;

    
    int _dim_dense;
    
    int _dim_sparse;

    
    unsigned int _ntrees;
    
    unsigned int _nthreads;

    
    int _train_loss;
  public:

    
    DecisionForest() :   _dim_dense(0), _dim_sparse(0),
			 _ntrees(0), _nthreads(1), _train_loss(TrainLoss::INVALID) {}


    
    int train_loss() {
      return _train_loss;
    }
    
    
    DecisionTree<d_t,i_t,v_t> & operator [] (size_t i) {return _dtree_vec[i];}

    
    void set(int nthreads, int ntrees=0) {
      _ntrees=ntrees;
      _nthreads=nthreads;
    }
    
    virtual double apply(DataPoint<d_t,i_t,v_t> & dp) {
      return apply(dp, _ntrees,_nthreads);
    }

    
    double apply(DataPoint<d_t,i_t,v_t> & dp, unsigned int ntrees, int nthreads);

    
    size_t appendFeatures(DataPoint<d_t,i_t,v_t> & dp, vector<int> & feat_vec, size_t offset);

    
    void write(ostream & os);

    
    void read(istream & is);

    
    void clear() {
      _dtree_vec.clear();
    }

    ~DecisionForest() {
      clear();
    }

    
    class TrainParam: public ParameterParser {
    public:

      
      ParamValue<string> loss;

      
      ParamValue<float> step_size;

      
      
      ParamValue<string> opt;
      
      
      ParamValue<int> ntrees;

      
      ParamValue<int> eval_frequency;
      
      
      ParamValue<int> write_frequency;

      
      ParamValue<int> verbose;
	  
      
      TrainParam(string prefix = "rgf.") {
	step_size.insert(prefix + "stepsize", 0, "step size of epsilon-greedy boosting (inactive for rgf)",
			 this,false);
	opt.insert(prefix + "opt", "rgf", "optimization method for training forest (rgf or epsilon-greedy)",
		   this);
	ntrees.insert(prefix + "ntrees", 500, "number of trees",
		      this);
	eval_frequency.insert(prefix+ "eval_frequency",50, "evaluate performance on test data at this frequency",this);
	write_frequency.insert(prefix+ "save_frequency",0, "save forest models to file 'model_file-iter' at this frequency",this);
      }
    };

    
    void train(
	       DataSet<d_t,i_t,v_t> & ds, double* scr_arr, 
	       class DecisionTree<d_t,i_t,v_t>::TrainParam &param_dt,
	       TrainParam & param_forest,
	       DataSet<d_t,i_t,v_t> & tst,
	       string model_file="",
	       DataDiscretizationInt *disc_ptr=0);

    
    void revert_discretization(DataDiscretizationInt & disc);

    
    void print(ostream & os, vector<string> & feature_names,
	       bool depth_first=true);
  };

  using DecisionForestFlt=DecisionForest<float,src_index_t,float>;
  using DecisionForestInt=DecisionForest<int,int,int>;
  using DecisionForestShort=DecisionForest<DISC_TYPE_T>;

}

#endif
