/************************************************************************
 *  dtree.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _RGF_DTREE_H

#define _RGF_DTREE_H

#include "classifier.h"
#include "discretization.h"

namespace rgf {
  

  class TreeNode {
  public:
    
    int_t feature;

    
    int_t sparse_index;
    
    
    double cut;

    
    double prediction;

    
    int left_index, right_index;

    
  TreeNode() :
    feature(-1), sparse_index(0),
    cut(0), prediction(0.0), left_index(-1), right_index(-1) {
    }

    
    template<typename d_t, typename i_t, typename v_t>
    inline int nextNodeIndex(DataPoint<d_t,i_t,v_t> & dp, bool is_sorted)
    {
      if (feature<0 || feature>= dp.dim_dense+dp.dim_sparse
	  || (left_index<0 && right_index<0)) return -1;

      
      if (feature<dp.dim_dense) {
	return (dp.x_dense[feature] <= cut)? left_index:right_index;
      }
      
      auto tmp=&dp.x_sparse[feature];
      if (! is_sorted) {
	for (int j=0; j<tmp->size(); j++) {
	  if ((*tmp)[j].index==sparse_index)
	    return ((*tmp)[j].value <= cut)? left_index:right_index;
	}
      }
      else { 
	int_t b=0;
	int_t e=dp.x_sparse[feature].size();
	while (e>b) {
	  int_t j=(e+b)/2;
	  if ((*tmp)[j].index>sparse_index) {
	    e=j;
	  }
	  else {
	    if ((*tmp)[j].index==sparse_index) {
	      return ((*tmp)[j].value <= cut)? left_index:right_index;
	    }
	    else b=j+1;
	  }
	}
      }
      
      
      return left_index;
    }

    
    bool is_leaf() {
      return (left_index < 0 && right_index<0);
    }

    
    void write(ostream & os);
    
    void read(istream & is);

    
    void clear() {
      feature = -1;
      cut = 0;
      left_index = right_index = -1;
    }
  };

  class DecisionForestTrainer;
  
  
  template<typename d_t, typename i_t, typename v_t>
  class DecisionTree: public BinaryClassifier<d_t,i_t,v_t> {
    
    vector<TreeNode> _nodes_vec;

    
    int _root_index;

    friend class DecisionForestTrainer;
  public:

    
  DecisionTree() :  _root_index(-1) {
    }

    
    int root() { return _root_index; }
    
    int size() {return _nodes_vec.size();}
    
    TreeNode & operator [] (int i) {return _nodes_vec[i];}
    
    
    inline int leaf_node_index(DataPoint<d_t,i_t,v_t> & dp, bool is_sorted)
    {
      int current_index=_root_index;
      while (current_index>=0) {
	int next_index = _nodes_vec[current_index].nextNodeIndex(dp,is_sorted);
	if (next_index <0) return current_index;
	current_index=next_index;
      }
      return current_index;
    }

    
    virtual double apply(DataPoint<d_t,i_t,v_t> & dp) {
      return apply(dp, false);
    }

    
    double apply(DataPoint<d_t,i_t,v_t> & dp, bool is_sorted) {
      int leaf_index=leaf_node_index(dp,is_sorted);
      return leaf_index>=0?_nodes_vec[leaf_index].prediction:0;
    }

    
    size_t appendFeatures(DataPoint<d_t,i_t,v_t> & dp,  vector<int> & feat_vec, size_t offset, bool is_sorted=true);

    
    size_t numFeatures() {
      return _nodes_vec.size();
    }

    
    void write(ostream & os);

    
    void read(istream & is);

    
    void clear() {
      _nodes_vec.clear();
      _root_index = -1;
    }

    ~DecisionTree() {
      clear();
    }

    
    class TrainParam: public ParameterParser {
    public:
      
      ParamValue<string> loss;

      
      ParamValue<int> maxLev;

      
      ParamValue<int> maxNodes;

      
      ParamValue<float> newTreeGainRatio;

      
      ParamValue<int> min_sample;

      
      ParamValue<float> lamL1;
      
      ParamValue<float> lamL2;

      
      ParamValue<int> nthreads;

      
      TrainParam(string prefix = "dt.") {
	loss.insert(prefix + "loss", "LS", "loss (LS or MODLS or LOGISTIC)", this);
	maxLev.insert(prefix + "max_level", 6, "maximum level of the tree",
		      this);
	maxNodes.insert(prefix + "max_nodes", 50,
			"maximum number of leaf nodes in best-first search", this);
	newTreeGainRatio.insert(prefix + "new_tree_gain_ratio",1.0,
				"new tree is created when leaf-nodes gain < this value * estimated gain of creating new three", this);
	min_sample.insert(prefix + "min_sample", 5,
			  "minum sample per node", this);

	lamL1.insert(prefix + "lamL1", 1,
		     "L1 regularization parameter", this);

	lamL2.insert(prefix + "lamL2", 1000,
		     "L2 regularization parameter", this);

	
	
      }

    };

    
    void train(DataSet<d_t,i_t,v_t> & ds,
	       double* scr_arr, 
	       TrainParam & param_dt);

    
    void revert_discretization(DataDiscretizationInt & disc);

    
    void print(ostream & os, int dim_dense, int dim_sparse, 
	       vector<string> & feature_names, bool depth_first=true);
    
  };

  using DecisionTreeFlt=DecisionTree<float,src_index_t,float>;
  using DecisionTreeInt=DecisionTree<int,int,int>;
  using DecisionTreeShort=DecisionTree<DISC_TYPE_T>;
  
}

#endif
