/************************************************************************
 *  node_trainer.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _DECISIONTREE_TRAINER_NODETRAINER_H

#define _DECISIONTREE_TRAINER_NODETRAINER_H

#include "dtree.h"
#include "training_target.h"
#include "feature_mapper.h"



namespace _decisionTreeTrainer
{
 
  template<typename d_t, typename i_t, typename v_t>
  class NodeTrainer {
    
    class GainThreadOutput {
    public:
      
      int feature;
      
      int sparse_index;
      
      int cut;
      
      double gain;

      
      double prediction;
      
      double left_prediction;
      
      double right_prediction;

      
      int yw_sum_arr_size=0;
      unique_ptr<YW_struct []> yw_sum_arr;
      
      GainThreadOutput()
      {
	gain=-1e10;
	cut=0;
      }
    };

    
    class ComputeGainMR : public MapReduce {
    public:
      
      vector<GainThreadOutput> gs;
      
      class DecisionTree<d_t,i_t,v_t>::TrainParam * param_dt_ptr=nullptr;

      
      NodeTrainer * node_ptr;

      
      FeatureMapper * fmapper_ptr;
      
      int dim_dense;
      
      int best_tid;
      
      ComputeGainMR(int nthreads,
		    class DecisionTree<d_t,i_t,v_t>::TrainParam &param_dt,
		    NodeTrainer *_node_ptr,
		    FeatureMapper & fmapper)
      {
	param_dt_ptr = &param_dt;
	node_ptr = _node_ptr;
	fmapper_ptr=& fmapper;
	dim_dense=node_ptr->featmap_dense.size();
	
	gs.resize(nthreads+1);
	best_tid=nthreads;
	for (int tid=0; tid<gs.size(); tid++) {
	  gs[tid].prediction=node_ptr->prediction;
	  gs[tid].gain=node_ptr->gain;
	  gs[tid].feature=0;
	  gs[tid].sparse_index=0;
	  gs[tid].cut=-1;
	  gs[tid].left_prediction=node_ptr->left_prediction;
	  gs[tid].right_prediction=node_ptr->right_prediction;
	}
      }

      void map(int tid, int j) {
	int my_feat;
	int my_index;
	int my_cut;
	double my_gain;
	double my_left_prediction;
	double my_right_prediction;
	double my_prediction;
	int dim_sparse=node_ptr->dim_sparse;
	assert(j>=0 && j<dim_dense+dim_sparse);
	if (j>= dim_sparse) { 
	  my_feat= j-dim_sparse;
	  int max_size=node_ptr->featmap_dense[my_feat].size+1;
	  if (gs[tid].yw_sum_arr_size<= max_size) {
	    gs[tid].yw_sum_arr_size= 2* max_size+1;
	    gs[tid].yw_sum_arr.reset(new YW_struct [2*max_size+1]);
	  }
	  node_ptr->featmap_dense[my_feat].compute_gainLS
	    ( my_cut, my_gain,
	      my_prediction,
	      my_left_prediction, my_right_prediction,
	      param_dt_ptr->min_sample.value, param_dt_ptr->lamL1.value, param_dt_ptr->lamL2.value,
	      node_ptr->data_size, 
	      node_ptr->data_dense_start+node_ptr->nrows*my_feat,
	      node_ptr->target.yw_data_arr(),node_ptr->target.prediction0(),
	      gs[tid].yw_sum_arr.get());
	  if (gs[tid].gain+1e-6 < my_gain) { 
	    gs[tid].feature=my_feat;
	    gs[tid].gain=my_gain;
	    gs[tid].cut=my_cut;
	    gs[tid].prediction=my_prediction;
	    gs[tid].left_prediction=my_left_prediction;
	    gs[tid].right_prediction=my_right_prediction;
	  }
	}
	else { 
	  my_feat= j;
	  int max_size=node_ptr->featmap_sparse[my_feat].offset[node_ptr->featmap_sparse[my_feat].nfeats]+1;
	  if (gs[tid].yw_sum_arr_size<= max_size) {
	    gs[tid].yw_sum_arr_size= 2* max_size+1;
	    gs[tid].yw_sum_arr.reset(new YW_struct [2*max_size+1]);
	  }
	  node_ptr->featmap_sparse[my_feat].compute_gainLS
	    (my_index, my_cut, my_gain,
	     my_prediction,
	     my_left_prediction, my_right_prediction,
	     param_dt_ptr->min_sample.value, param_dt_ptr->lamL1.value, param_dt_ptr->lamL2.value,
	     node_ptr->data_size, 
	     node_ptr->data_sparse_start+node_ptr->nrows*my_feat,
	     node_ptr->target.yw_data_arr(), node_ptr->target.prediction0(),
	     gs[tid].yw_sum_arr.get());
	  if ((gs[tid].gain+1e-6 < my_gain)
	      || (fmapper_ptr->is_valid && (gs[tid].gain-1e-6 <= my_gain)
		  && fmapper_ptr->to_original(gs[tid].feature,gs[tid].sparse_index)
		  > fmapper_ptr->to_original(my_feat+dim_dense,my_index)))
	    {
	      gs[tid].feature=my_feat+dim_dense;
	      gs[tid].sparse_index=my_index;
	      gs[tid].gain=my_gain;
	      gs[tid].cut=my_cut;
	      gs[tid].prediction=my_prediction;
	      gs[tid].left_prediction=my_left_prediction;
	      gs[tid].right_prediction=my_right_prediction;
	    }
	}
	return;
      }

      void reduce(int tid) {
	int f_i;
	int f_bi;
	if (fmapper_ptr->is_valid) {
	  f_i=fmapper_ptr->to_original(gs[tid].feature,gs[tid].sparse_index);
	  f_bi=fmapper_ptr->to_original(gs[best_tid].feature,gs[best_tid].sparse_index);
	}
	else {
	  f_i=gs[tid].feature;
	  f_bi=gs[best_tid].feature;
	  if (f_i>= dim_dense && f_i==f_bi) {
	    f_i=gs[tid].sparse_index;
	    f_bi=gs[best_tid].sparse_index;
	  }
	}
	if ((gs[tid].gain>gs[best_tid].gain+1e-6) || (gs[tid].gain>=gs[best_tid].gain-1e-6 && f_i<f_bi))
	  best_tid=tid;
      }
    };

  public:
    
    double prediction=0;

    
    int left_index=-1;
    
    int right_index=-1;

    
    int cut=0;

    
    int feature=0;

    
    int sparse_index=-1;

    
    double gain=0;

    
    double cut_orig=0;

    
    int level=0;
    
    double left_prediction=0;
    
    double right_prediction=0;

    
    UniqueArray<FeatureValueMapDense<d_t> > featmap_dense;

    
    int dim_sparse;
    FeatureValueMapSparse *featmap_sparse;

    
    size_t nrows=0;

    
    size_t data_size=0;

    
    TrainTarget target;
    
    
    d_t *data_dense_start=0;

    
    SparseFeatureElementArray<i_t,v_t> *data_sparse_start=0;


    NodeTrainer() {}
    
    NodeTrainer(TrainTarget  _target,
		d_t *_data_dense_start, int _dim_dense,
		SparseFeatureElementArray<i_t,v_t> *_data_sparse_start, int _dim_sparse,
		FeatureValueMapSparse *_featmap_sparse,
		double _prediction, size_t _data_size, size_t _nrows,  int _level) : 
      prediction(_prediction), left_index(-1), right_index(-1),
      level(_level),
      dim_sparse(_dim_sparse), 
      featmap_sparse(_featmap_sparse),
      nrows(_nrows),      data_size(_data_size),
      target(_target),
      data_dense_start(_data_dense_start),  data_sparse_start(_data_sparse_start)
    {
      featmap_dense.reset(_dim_dense);
    }

    bool is_leaf() {
      return (left_index<0) || (right_index<0);
    }

    
    void clear() {
      featmap_dense.reset(0);
      left_index=-1;
      right_index=-1;
    }

    
    ~NodeTrainer() {}

    
    void compute_gain(class DecisionTree<d_t,i_t,v_t>::TrainParam & param_dt,
		      FeatureMapper &fmapper) {

      MapReduceRunner runner(param_dt.nthreads.value,MapReduceRunner::INTERLEAVE);

      target.compute_yw(data_size,prediction,runner.nthreads);
	    
      ComputeGainMR mr(runner.nthreads,param_dt,this,fmapper);
      runner.run(mr,0,featmap_dense.size()+dim_sparse);
      
      feature=mr.gs[mr.best_tid].feature;
      sparse_index=mr.gs[mr.best_tid].sparse_index;
      gain=mr.gs[mr.best_tid].gain;
      cut=mr.gs[mr.best_tid].cut;
      prediction=mr.gs[mr.best_tid].prediction;
      left_prediction=mr.gs[mr.best_tid].left_prediction;
      right_prediction=mr.gs[mr.best_tid].right_prediction;

      cut_orig=-1;
      if (gain>=0 && feature >=0 && cut >=0) {
	if (feature<featmap_dense.size()) {
	  cut_orig=featmap_dense[feature].get(cut); 
	}
	else cut_orig=cut;  
      }
    }


    
    void split(vector<NodeTrainer<d_t,i_t,v_t> *> & node_vec, int nthreads) {

      if (gain<=0 || feature<0) return;
      pair<train_size_t,train_size_t> * swap_pairs=0;
      train_size_t left_size=0;
      train_size_t swap_size=0;
      
      int dim_dense=featmap_dense.size();

      MapReduceRunner runner(nthreads,MapReduceRunner::BLOCK);
      
      if (feature<dim_dense) {
	d_t *p=data_dense_start + nrows*feature; 
	
	if (data_size<runner.nthreads*10) {
	  for (train_size_t i=0; i<data_size; i++) {
	    if (p[i]<= cut) left_size++;
	  }
	}
	else {
	  class CutCountMR : public MapReduceCounter<int> {
	  public:
	    d_t *p;
	    int cut;
	  public:
	    void map_range(int tid, int b, int e) {
	      int result=0;
	      for (int i=b; i<e; i++) if (p[i]<=cut) result++;
	      counts[tid]=result;
	    }
	  } mr;
	  mr.set_nthreads(runner.nthreads,0);
	  mr.p=p;
	  mr.cut=cut;
	  runner.run_range(mr,0,data_size);
	  left_size=mr.result;
	}
	swap_size=min((int_t) left_size,(int_t)(data_size-left_size));
	if (swap_size>0) swap_pairs= new pair<train_size_t,train_size_t> [swap_size];
	swap_size=0;
	
	train_size_t left_i=0;
	train_size_t right_i;
	for (right_i=left_size; right_i<data_size; right_i++) {
	  if(p[right_i]>cut) continue;
	  while(p[left_i]<=cut) left_i++; 
	  swap_pairs[swap_size++]=pair<train_size_t,train_size_t>(left_i,right_i);
	  left_i++;
	}
      }
      else {
	bool *vle = new bool [data_size];
	SparseFeatureElementArray<i_t,v_t> *p=data_sparse_start + nrows*(feature-dim_dense);
	class SparseCutMR : public MapReduce {
	public:
	  SparseFeatureElementArray<i_t,v_t> *p;
	  bool *vle;
	  int_t sparse_index;
	  int_t cut;
	  inline void map(int tid, int j) {
	    vle[j]= (p[j].value_less_or_equal(sparse_index,cut));
	  }
	} mr;
	mr.p=p;
	mr.vle=vle;
	mr.sparse_index=sparse_index;
	mr.cut=cut;
	runner.run(mr,0,data_size);
	for (train_size_t i=0; i<data_size; i++) {
	  if (vle[i]) left_size++;
	}
	swap_size=min((int_t)left_size,(int_t)(data_size-left_size));
	if (swap_size>0) swap_pairs= new pair<train_size_t,train_size_t> [swap_size];
	swap_size=0;
	
	train_size_t left_i=0;
	train_size_t right_i;
	for (right_i=left_size; right_i<data_size; right_i++) {
	  if(!vle[right_i]) continue;
	  while(vle[left_i]) left_i++; 
	  swap_pairs[swap_size++]=pair<int,int>(left_i,right_i);
	  left_i++;
	}
	delete [] vle;
      }

      runner.set(nthreads,MapReduceRunner::INTERLEAVE);
      class SwapFeatMR : public MapReduce {
      public:
	
	size_t nrows;
	
	
	int dim_dense;
      
	
	d_t *data_dense_start;

	
	int dim_sparse;
      
	
	SparseFeatureElementArray<i_t,v_t>  *data_sparse_start;

	
	TrainTarget *target_ptr;
      
	
	train_size_t swap_size;
	pair<train_size_t,train_size_t> *swap_pairs;

	void map(int tid, int j) {
	  if (j < dim_dense) {
	    swap_arr<d_t>(data_dense_start+nrows*j, swap_pairs,swap_size);
	  }
	  else {
	    int my_feat= j-dim_dense;
	    if (my_feat< dim_sparse) {
	      swap_arr<SparseFeatureElementArray<i_t,v_t> >(data_sparse_start+nrows*my_feat, swap_pairs,swap_size);
	    }
	    else {
	      assert(my_feat==dim_sparse);
	      target_ptr->swap(swap_pairs,swap_size);
	    }
	  }
	}
      } mr;
      mr.nrows=nrows;
      mr.dim_dense=dim_dense;
      mr.data_dense_start=data_dense_start;
      mr.dim_sparse=dim_sparse;
      mr.data_sparse_start=data_sparse_start;
      mr.swap_size=swap_size;
      mr.swap_pairs=swap_pairs;
      mr.target_ptr=&target;
      runner.run(mr,0,dim_dense+dim_sparse+1);
      
      delete [] swap_pairs;

      
      
      left_index=node_vec.size();
      NodeTrainer<d_t,i_t,v_t> * node_left
	= new NodeTrainer<d_t,i_t,v_t>(target,
				       data_dense_start, featmap_dense.size(),
				       data_sparse_start,dim_sparse, featmap_sparse,
				       left_prediction,
				       left_size, nrows, level+1);
      node_vec.push_back(node_left);
      
      
      right_index=node_vec.size();
      NodeTrainer<d_t,i_t,v_t> *node_right
	= new NodeTrainer<d_t,i_t,v_t>(target.shift(left_size),  
				       data_dense_start+left_size, featmap_dense.size(),
				       data_sparse_start+left_size,dim_sparse, featmap_sparse,
				       right_prediction,
				       data_size-left_size,nrows,level+1);
      node_vec.push_back(node_right);

      
      
      {
    	  int my_feat;
    	  int max_fmap_size=0;
    	  for (my_feat=0; my_feat<featmap_dense.size(); my_feat++) {
	    max_fmap_size=max(max_fmap_size,featmap_dense[my_feat].size);
    	  }

    	  d_t * tmp_arr= new d_t [(max_fmap_size+1)*2];
    	  for (my_feat=0; my_feat<featmap_dense.size(); my_feat++) {
	    
	    node_vec[left_index]->featmap_dense[my_feat].initFrom
	      (featmap_dense[my_feat], false,
	       data_dense_start+nrows*my_feat,
	       left_size,
	       tmp_arr);
	    
	    node_vec[right_index]->featmap_dense[my_feat].initFrom
	      (featmap_dense[my_feat], false,
	       data_dense_start+left_size+nrows*my_feat,
	       data_size-left_size,tmp_arr);
    	  }
    	  delete [] tmp_arr;
      }
    }

  };
}

#endif
