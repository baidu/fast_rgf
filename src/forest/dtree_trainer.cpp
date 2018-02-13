/************************************************************************
 *  dtree_trainer.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "dtree.h"
#include "node_trainer.h"
#include "feature_mapper.h"

static Timer t0("in dtree training: time for data initializing in each tree training");
static Timer t1("in dtree training: time for finding best node splits (multi-thread)");
static Timer t2("in dtree training: time for doing node splits (partial-multi-thread)");
static Timer t3("in fully-corrective-update: time for computing y-w statistics");
static Timer t4("in fully-corrective-update: time for updating predictions");


namespace _decisionTreeTrainer
{
  
  template<typename d_t, typename i_t, typename v_t>
  class SingleTreeTrainer {
  public:
    
    vector<NodeTrainer<d_t,i_t,v_t> *> node_vec;
    
    
    int root_index;

    
    FeatureMapper fmapper;
    
    SingleTreeTrainer() : root_index(-1) {}

    
    void copyTo(NodeTrainer<d_t,i_t,v_t> & node_from, TreeNode & node_to) {
      if (fmapper.is_valid) {
	node_to.feature=0;
	node_to.sparse_index=fmapper.to_original(node_from.feature,node_from.sparse_index);
      }
      else {
	node_to.feature=node_from.feature;
	node_to.sparse_index=node_from.sparse_index;
      }
      node_to.cut=node_from.cut_orig; 
      node_to.prediction=node_from.prediction;
      node_to.left_index=node_from.left_index;
      node_to.right_index=node_from.right_index;
    }

    
    void copyTo(vector<TreeNode> & tree_nodes)
    {
      tree_nodes.resize(node_vec.size());
      for (int i=0; i<tree_nodes.size(); i++) {
	copyTo(*node_vec[i],tree_nodes[i]);
      }
    }

    
    void clear_nodes() {
      for (int i=0; i<node_vec.size(); i++) {
	node_vec[i]->clear();
	delete node_vec[i];
      }
      node_vec.clear();
      root_index=-1;
    }
    
    ~SingleTreeTrainer() {}

    
    
    void train(DataSet<d_t,i_t,v_t> & ds, double *scr_arr, class DecisionTree<d_t,i_t,v_t>::TrainParam & tr, double step_size);

    
    
    size_t nrows;

    TrainTarget target;
    
    
    int dim_dense;
    
    int dim_sparse;


    
    d_t * data_dense;
    d_t * data_dense2;
    
    
    d_t * maxval_dense;


    
    FeatureValueMapSparse *featmap_sparse;
    
    
    SparseFeatureElementArray<i_t,v_t> * data_sparse=nullptr;
    int * data_sparse_size=nullptr;
    
    SparseFeatureElement<i_t,v_t> * data_sparse_storage=nullptr;

    
    size_t * elem_size_offset_arr=nullptr;
    

    
    void init_with_fmapper(DataSet<d_t,i_t,v_t> & ds)
    {
      size_t i;

      nrows=ds.size();
      dim_dense=fmapper.dim_dense;

      maxval_dense = new d_t [dim_dense];
      memset(maxval_dense,0,sizeof(d_t)*(size_t) dim_dense);
      data_dense= new d_t [nrows *(size_t) dim_dense];
      data_dense2=new d_t [nrows *(size_t) dim_dense];
      memset(data_dense2,0,sizeof(d_t)*(size_t)nrows*(size_t)dim_dense);

      dim_sparse=fmapper.dim_sparse;
      featmap_sparse = new FeatureValueMapSparse [dim_sparse];
      for (int j=0; j< dim_sparse; j++) {
	featmap_sparse[j].reset(fmapper.sparse_to_original[j].size());
      }

      
      data_sparse= new SparseFeatureElementArray<i_t,v_t> [dim_sparse*nrows];
      data_sparse_size= new int [dim_sparse*(size_t)nrows];
      memset(data_sparse_size,0,sizeof(int)*nrows*(size_t)dim_sparse);
      int *data_sparse_start= new int [dim_sparse*(size_t)nrows];
      size_t elem_size=0;
      elem_size_offset_arr = new size_t [nrows];

      for (i=0; i<nrows; i++) {
	elem_size_offset_arr[i] = elem_size;
	SparseFeatureGroup<i_t,v_t> * x_sparse = ds.x_sparse[i];
	int j=0;
	for (int k=0; k<x_sparse[j].size(); k++) {
	  SparseFeatureElement<i_t,v_t> elem=x_sparse[j][k];
	  int jj=elem.index;
	  pair<int,int> ff=fmapper.original_to_feature[jj];
	  if (ff.first<0) { 
	    data_dense2[i+ff.second*nrows]=elem.value;
	    if (maxval_dense[ff.second]<elem.value) maxval_dense[ff.second]=elem.value;
	  }
	  else { 
	    data_sparse_size[i+ff.first*(size_t)nrows]++;
	    if (featmap_sparse[ff.first].offset[1+ff.second]<elem.value)
	      featmap_sparse[ff.first].offset[1+ff.second]=elem.value;
	  }
	}
	for (j=0; j<dim_sparse; j++) {
	  data_sparse_start[i+j*nrows]=elem_size;
	  elem_size+=data_sparse_size[i+j*nrows];
	}
      }

      for (int j=0; j<dim_sparse; j++) {
	for (int k=0; k<featmap_sparse[j].nfeats; k++) {
	  featmap_sparse[j].offset[k+1] += featmap_sparse[j].offset[k]+1;
	}
      }

      
      data_sparse_storage = new SparseFeatureElement<i_t,v_t> [elem_size];
      for (i=0; i<nrows; i++) {
	SparseFeatureGroup<i_t,v_t> * x_sparse = ds.x_sparse[i];
	int j=0;
	for (int k=0; k<x_sparse[j].size(); k++) {
	  SparseFeatureElement<i_t,v_t> elem=x_sparse[j][k];
	  int jj=elem.index;
	  pair<int,int> ff=fmapper.original_to_feature[jj];
	  if (ff.first>=0) { 
	    data_sparse_storage[data_sparse_start[i+(size_t)ff.first*(size_t)nrows]++]
	      =SparseFeatureElement<i_t,v_t>(ff.second,elem.value);
	  }
	}
      }
      delete [] data_sparse_start;

    }
	
    
    void init(DataSet<d_t,i_t,v_t> & ds, int ngrps, int verbose)
    {
      fmapper.init(ds,ngrps);
      nrows=ds.size();


      if (fmapper.is_valid) {
	init_with_fmapper(ds);

	if (verbose>=5) {
	  cerr << "  split features into " << fmapper.dim_dense << " dense features"
	       << " and " << fmapper.dim_sparse << " sparse feature groups" <<endl;
	  fflush(stderr);
	}
	return;
      }

      dim_dense=ds.dim_dense();
      
      maxval_dense = new d_t [dim_dense];
      memset(maxval_dense,0,sizeof(d_t)*(size_t)dim_dense);

      data_dense= new d_t [nrows *(size_t)dim_dense];
      data_dense2=new d_t [nrows *(size_t)dim_dense];
      size_t i;
      for (i=0; i<nrows; i++) {
	int j;
	d_t *v_ptr=ds.x_dense[i];
	for (j=0; j<dim_dense; j++) {
	  d_t v=v_ptr[j];
	  data_dense2[i+j*(size_t)nrows]=v;
	  if (maxval_dense[j]<v) maxval_dense[j]=v;
	}
      }

      
      dim_sparse=ds.dim_sparse();
      featmap_sparse = new FeatureValueMapSparse [dim_sparse];

      int elem_size=0;
      
      if (dim_sparse>0) {
	elem_size_offset_arr = new size_t [nrows];
	for (i=0; i<nrows; i++) {
	  elem_size_offset_arr[i]=elem_size;
	  SparseFeatureGroup<i_t,v_t> * x_sparse = ds.x_sparse[i];
	  for (int j=0; j<dim_sparse; j++) {
	    elem_size += x_sparse[j].size();
	    for (int k=0; k<x_sparse[j].size(); k++) 
	      featmap_sparse[j].nfeats=max(featmap_sparse[j].nfeats,x_sparse[j][k].index);
	  }
	}
      }
      
      for (int j=0; j< dim_sparse; j++) {
	featmap_sparse[j].reset(featmap_sparse[j].nfeats+1);
      }
      
      data_sparse_storage = new SparseFeatureElement<i_t,v_t> [elem_size];
      
      data_sparse= new SparseFeatureElementArray<i_t,v_t> [dim_sparse*(size_t)nrows];
      
      if (dim_sparse>0) {
	elem_size=0;
	for (i=0; i<nrows; i++) {
	  SparseFeatureGroup<i_t,v_t> * x_sparse = ds.x_sparse[i];
	  for (int j=0; j<dim_sparse; j++) {
	    memcpy(data_sparse_storage+elem_size, x_sparse[j].get(),
		   x_sparse[j].size()*sizeof(SparseFeatureElement<i_t,v_t>));
	    for (int k=0; k<x_sparse[j].size(); k++) {
	      SparseFeatureElement<i_t,v_t> elem=x_sparse[j][k];
	      if (featmap_sparse[j].offset[1+elem.index]< elem.value)
		featmap_sparse[j].offset[1+elem.index]=elem.value;
	    }
	    elem_size += x_sparse[j].size();
	  }
	}
      }
      for (int j=0; j<dim_sparse; j++) {
	for (int k=0; k<featmap_sparse[j].nfeats; k++) {
	  featmap_sparse[j].offset[k+1] += featmap_sparse[j].offset[k]+1;
	}
      }

    }

    
    void finish(int verbose)
    {
      clear_nodes();

      target.clear();
      
      delete [] data_dense;
      delete [] data_dense2;
      delete [] maxval_dense;

      delete [] featmap_sparse;

      delete [] data_sparse;
      delete [] data_sparse_size;
      delete [] data_sparse_storage;

      delete [] elem_size_offset_arr;

      if (verbose>=5) {
	t0.print();
	t1.print();
	t2.print();
      }
    }
    
  };


} 


using namespace _decisionTreeTrainer;


template<typename d_t, typename i_t, typename v_t>
void SingleTreeTrainer<d_t,i_t,v_t>::train
(DataSet<d_t,i_t,v_t> & ds, double *scr_arr, class DecisionTree<d_t,i_t,v_t>::TrainParam & param_dt,
 double step_size)
{
  clear_nodes();
  if (nrows<=0 || dim_dense+dim_sparse<=0) return;

  t0.start();
  
  
  target.set(nrows, ds.y.data(), scr_arr,
	     (ds.row_weights.size()>0)?ds.row_weights.data():nullptr,
	     param_dt.loss.value, ds.y_type);
  
  memcpy(data_dense,data_dense2,nrows*dim_dense*sizeof(d_t));
  
  if (dim_sparse>0) {
    MapReduceRunner runner(param_dt.nthreads.value,MapReduceRunner::BLOCK);
    class SparseDataCpyMR : public MapReduce {
    public:
      DataSet<d_t,i_t,v_t> * ds_ptr;
      SingleTreeTrainer<d_t,i_t,v_t> * trainer_ptr;

      void map(int tid, int i) {
	train_size_t nrows=trainer_ptr->nrows;
	SparseFeatureGroup<i_t,v_t> * x_sparse = ds_ptr->x_sparse[i];
	size_t elem_size=trainer_ptr->elem_size_offset_arr[i];
	if (trainer_ptr->fmapper.is_valid) {
	  for (int j=0; j<trainer_ptr->dim_sparse; j++) {
	    int mysize=trainer_ptr->data_sparse_size[i+j*nrows];
	    trainer_ptr->data_sparse[i+j*nrows]=
	      SparseFeatureElementArray<i_t,v_t>(mysize,trainer_ptr->data_sparse_storage+elem_size);
	    elem_size +=mysize;
	  }
	}
	else {
	  for (int j=0; j<trainer_ptr->dim_sparse; j++) {
	    int mysize=x_sparse[j].size();
	    trainer_ptr->data_sparse[i+j*nrows]=
	      SparseFeatureElementArray<i_t,v_t>(mysize,trainer_ptr->data_sparse_storage+elem_size);
	    elem_size +=mysize;
	  }
	}
	return;
      }
    } mr;
      
    mr.ds_ptr=&ds;
    mr.trainer_ptr= this;
    runner.run(mr,0,nrows);
  }
  t0.stop();
  
  class QuElem {
  public:
    
    double gain;
    
    int index;
    
    const bool operator< (const QuElem & b) const {
      return (gain < b.gain);
    }
    QuElem(int _index,  double _gain) : gain(_gain), index(_index) {}
  };


  
  priority_queue<QuElem,vector<QuElem> > qu;
    
  int index;
  
  
  double gain0, gain_leaves;
  {

    
    NodeTrainer<d_t,i_t,v_t> * root_node = new NodeTrainer<d_t,i_t,v_t>(target,
									data_dense, dim_dense,
									data_sparse, dim_sparse,
									featmap_sparse,
									0,nrows,nrows,0);
    
    root_node->featmap_dense.reset(dim_dense);
    for (int j=0; j<dim_dense; j++) {
      root_node->featmap_dense[j].initFrom(1+(int)maxval_dense[j]);
    }
    root_node->prediction=0;
    t1.start();
    root_node->compute_gain(param_dt,fmapper);
    t1.stop();
    
    index=root_index=0;
    node_vec.push_back(root_node);

    if (node_vec[index]->level<param_dt.maxLev.value) {
      qu.push(QuElem(index,node_vec[index]->gain));
      gain_leaves=node_vec[index]->gain;
      gain0=param_dt.newTreeGainRatio.value*gain_leaves;
    }
  }
  
  bool split_root=true;
  while (qu.size() < param_dt.maxNodes.value && qu.size()>0) {
    QuElem current=qu.top();
    qu.pop();
    if (gain_leaves < gain0 && ! split_root) continue;
    split_root=false;
    gain_leaves -= current.gain;
    if (node_vec[current.index]->level>=param_dt.maxLev.value) continue;
    
    NodeTrainer<d_t,i_t,v_t> *node_ptr= node_vec[current.index];
    t2.start();
    node_ptr->split(node_vec,param_dt.nthreads.value);
    t2.stop();
    
    
    index=node_ptr->left_index;
    if (index<0) {
      assert(node_ptr->right_index<0);
      continue;
    }
    if (node_vec[index]->level>=param_dt.maxLev.value) {
      node_vec[index]->feature=0;
      node_vec[index]->gain=0;
    }
    else {
      t1.start();
      node_vec[index]->compute_gain(param_dt, fmapper);
      t1.stop();
    }
    qu.push(QuElem(index,node_vec[index]->gain));
    gain_leaves+=node_vec[index]->gain;
    
    index=node_ptr->right_index;
    assert(index>=0);
    if (node_vec[index]->level>=param_dt.maxLev.value) {
      node_vec[index]->feature=0;
      node_vec[index]->gain=0;
    }
    else {
      t1.start();
      node_vec[index]->compute_gain(param_dt, fmapper);
      t1.stop();
    }
    qu.push(QuElem(index,node_vec[index]->gain));
    gain_leaves+=node_vec[index]->gain;
  }

  
  if (scr_arr !=nullptr) {
    for (int i=0; i< node_vec.size(); i++) {
      node_vec[i]->prediction *=step_size;
      if (node_vec[i]->left_index>=0) continue;
      assert(node_vec[i]->right_index<0);
      double prediction=node_vec[i]->prediction;
      train_size_t * index=node_vec[i]->target.index;
      train_size_t data_size = node_vec[i]->data_size;
      for (train_size_t j=0; j<data_size; j++) {
	scr_arr[index[j]] += prediction;
      }
    }
  }
  return;
}

namespace _decisionTreeTrainer
{

  
  template<typename d_t, typename i_t, typename v_t>
  class TreeToIndex {
  public:
    
    int tree_id;

    
    struct NodeStruct {
    public:
      
      int node_id;
      
      double prediction;
      NodeStruct(int id, double p) : node_id(id), prediction(p) {}
    };
    vector<NodeStruct> nodes;
    
    
    UniqueArray<int> reverse_index;

    
    void set(SingleTreeTrainer<d_t,i_t,v_t> & trainer, int id)
    {
      tree_id=id;
      reverse_index.reset(trainer.nrows);
      for (int i=0; i<trainer.node_vec.size(); i++) {
	if (!trainer.node_vec[i]->is_leaf()) continue;
	int id=nodes.size();
	auto node_ptr=trainer.node_vec[i];
	nodes.push_back(NodeStruct(i,node_ptr->prediction));
	for (int k=0; k<node_ptr->data_size; k++) {
	  reverse_index[node_ptr->target.index[k]]=id;
	}
      }
      return;
    }

    
    void copyPredictionFrom(DecisionTree<d_t,i_t,v_t> & dtree) {
      for (int i=0; i<nodes.size(); i++) {
	nodes[i].prediction=dtree[nodes[i].node_id].prediction;
      }
    }

    
    void copyPredictionTo(DecisionTree<d_t,i_t,v_t> & dtree) {
      for (int i=0; i<nodes.size(); i++) {
	dtree[nodes[i].node_id].prediction=nodes[i].prediction;
      }
    }

    
    void update_predictions(TrainTarget & target, class DecisionTree<d_t,i_t,v_t>::TrainParam & param_dt, Timer & t1, Timer & t2)
    {
      if (nodes.size()==0) return;
      
      int nthreads=param_dt.nthreads.value;
      double lamL1=param_dt.lamL1.value;
      double lamL2=param_dt.lamL2.value;
      MapReduceRunner runner(nthreads,MapReduceRunner::BLOCK);
      
      t3.start();
      class Tree_YW_MR : public MapReduce {
      public:
	vector<YW_struct> yw_thread;
	vector<YW_struct> yw_result;
	TrainTarget * target_ptr;
	int * reverse_index;
	void map_range(int tid, int b, int e) {
	  target_ptr->compute_yw(reverse_index,b,e,&yw_thread[yw_result.size()*tid],yw_result.size());
	}
	void reduce(int tid) {
	  for (int j=0; j<yw_result.size(); j++) yw_result[j].add(yw_thread[yw_result.size()*tid+j]);
	}
      } mr1;
      runner.set(nthreads,MapReduceRunner::BLOCK);
      mr1.target_ptr=&target;
      mr1.reverse_index=reverse_index.get();
      mr1.yw_result.resize(nodes.size());
      for (int i=0; i<mr1.yw_result.size(); i++) mr1.yw_result[i].set(0,0);
      mr1.yw_thread.resize(mr1.yw_result.size()*runner.nthreads);
      runner.run_range(mr1,0,reverse_index.size());
      t3.stop();

      
      t4.start();
      class Tree_Update_MR : public MapReduce {
      public:
	TrainTarget * target_ptr;
	vector<double> delta_p;
	int * reverse_index;
	void map_range(int tid, size_t b, size_t e) {
	  for (int j=b; j<e; j++) target_ptr->residues[j] += delta_p[reverse_index[j]];
	}
      } mr2;
      mr2.target_ptr=&target;
      mr2.delta_p.resize(nodes.size());
      mr2.reverse_index=reverse_index.get();

      for (int i=0; i<nodes.size(); i++) {
	double x;
	double p0=nodes[i].prediction;
	solve_L1_L2(mr1.yw_result[i].w,p0*mr1.yw_result[i].w+mr1.yw_result[i].y, lamL1, lamL2,x);
	mr2.delta_p[i]=x-p0;
	nodes[i].prediction=x;
      }
      runner.set(nthreads,MapReduceRunner::BLOCK);
      runner.run_range(mr2,0,reverse_index.size());
      t4.stop();
    }
  };
  
  template<typename d_t, typename i_t, typename v_t>
  class MultiTreeTrainer {
  public:
    
    SingleTreeTrainer<d_t,i_t,v_t> dtree_trainer;
    vector<TreeToIndex<d_t,i_t,v_t> *> tree_vec;
    int cur_id=0;

    void add_leaf_nodes() {
      TreeToIndex<d_t,i_t,v_t> * tr = new TreeToIndex<d_t,i_t,v_t>;
      tr->set(dtree_trainer, tree_vec.size());
      tree_vec.push_back(tr);
    }

    ~MultiTreeTrainer() {
      for (int i=0; i<tree_vec.size(); i++) {
	delete tree_vec[i];
	tree_vec[i]=nullptr;
      }
      tree_vec.clear();
    }
  };
}



#include "forest_trainer.h"

template<typename d_t, typename i_t, typename v_t>
void DecisionForestTrainer::init(DataSet<d_t,i_t,v_t> & ds,int ngrps, int verbose)
{
  ds.sort();
  auto tmp =new (MultiTreeTrainer<d_t,i_t,v_t>);
  tmp->dtree_trainer.init(ds,ngrps,verbose);
  trainer_ptr = tmp;
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForestTrainer::finish(DataSet<d_t,i_t,v_t> &ds, int verbose)
{
  if (trainer_ptr) {
    auto tmp = (MultiTreeTrainer<d_t,i_t,v_t> *) trainer_ptr;
    tmp->dtree_trainer.finish(verbose);
    delete tmp;
    if (verbose>=5 && is_fully_corrective()) {
      t3.print();
      t4.print();
    }
    trainer_ptr=nullptr;
  }
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForestTrainer::build_single_tree(DataSet<d_t,i_t,v_t> & ds, double *scr_arr, 
					      class DecisionTree<d_t,i_t,v_t>::TrainParam & param_dt,
					      double step_size,
					      class DecisionTree<d_t,i_t,v_t> & dtree)
{
  MultiTreeTrainer<d_t,i_t,v_t> * my_trainer_ptr
    = (MultiTreeTrainer<d_t,i_t,v_t> *) trainer_ptr;

  if (is_fully_corrective()) step_size=1.0;
  my_trainer_ptr->dtree_trainer.train(ds,scr_arr,param_dt,step_size);

  my_trainer_ptr->dtree_trainer.copyTo(dtree._nodes_vec);
  dtree._root_index=my_trainer_ptr->dtree_trainer.root_index;
  if (is_fully_corrective()) {
    my_trainer_ptr->add_leaf_nodes();
  }
  my_trainer_ptr->dtree_trainer.clear_nodes();
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForestTrainer::fully_corrective_update(DataSet<d_t,i_t,v_t> & ds, double *scr_arr, 
						    class DecisionTree<d_t,i_t,v_t>::TrainParam &param_dt,
						    DecisionTree<d_t,i_t,v_t> * dtree_vec,
						    int ntrees)
{
  
  MultiTreeTrainer<d_t,i_t,v_t> * my_trainer_ptr
    = (MultiTreeTrainer<d_t,i_t,v_t> *) trainer_ptr;
  my_trainer_ptr->dtree_trainer.target.set(my_trainer_ptr->dtree_trainer.nrows, ds.y.data(), scr_arr,
					   (ds.row_weights.size()>0)?ds.row_weights.data():nullptr,
					   param_dt.loss.value, ds.y_type);

  assert(my_trainer_ptr->tree_vec.size()==ntrees);
  int i;
  
  int num_opts=80;
  
  int num_recent=20;
  if (ntrees<=num_recent) num_recent=ntrees;
  
  for (int it=0; it<num_opts; it++) {
    i= (my_trainer_ptr->cur_id++) % my_trainer_ptr->tree_vec.size();
    if (ntrees-num_recent<=i) continue;
    assert(my_trainer_ptr->tree_vec[i]->tree_id==i);
    my_trainer_ptr->tree_vec[i]->copyPredictionFrom(dtree_vec[i]);
    
    my_trainer_ptr->tree_vec[i]->update_predictions(my_trainer_ptr->dtree_trainer.target,param_dt, t1, t2);

    my_trainer_ptr->tree_vec[i]->copyPredictionTo(dtree_vec[i]);
  }
  for (i=ntrees-num_recent; i<ntrees; i++) {
    assert(my_trainer_ptr->tree_vec[i]->tree_id==i);
    my_trainer_ptr->tree_vec[i]->copyPredictionFrom(dtree_vec[i]);
    
    my_trainer_ptr->tree_vec[i]->update_predictions(my_trainer_ptr->dtree_trainer.target,param_dt, t1, t2);

    my_trainer_ptr->tree_vec[i]->copyPredictionTo(dtree_vec[i]);
  }
  my_trainer_ptr->dtree_trainer.target.copy_back(my_trainer_ptr->dtree_trainer.nrows, ds.y.data(), scr_arr);
}



namespace rgf {

  
  template void DecisionForestTrainer::init(DataSetShort&,int, int);
  template void DecisionForestTrainer::finish(DataSetShort&,int);
  template void DecisionForestTrainer::build_single_tree(DataSetShort&, double *, 
							 class DecisionTreeShort::TrainParam &,double,
							 DecisionTreeShort &);
  template void DecisionForestTrainer::fully_corrective_update(DataSetShort &, double *,
							       class DecisionTreeShort::TrainParam &,
							       DecisionTreeShort *,
							       int);

  
  template void DecisionForestTrainer::init(DataSetInt&,int, int);
  template void DecisionForestTrainer::finish(DataSetInt&,int);
  template void DecisionForestTrainer::build_single_tree(DataSetInt&, double *, 
							 class DecisionTreeInt::TrainParam &,double,
							 DecisionTreeInt &);
  template void DecisionForestTrainer::fully_corrective_update(DataSetInt &, double *,
							       class DecisionTreeInt::TrainParam &,
							       DecisionTreeInt *,
							       int);

}
