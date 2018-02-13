/************************************************************************
 *  forest.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "forest.h"
#include "utils.h"

static Timer t0("in forest training: time for initializing forest training");
static Timer t1("in forest training: time for tree training");
static Timer t2("in forest training: time for fully corrective update");
static Timer t3("in forest training: time for tree applying and evlauation");



template<typename d_t, typename i_t, typename v_t>
double DecisionForest<d_t,i_t,v_t>::apply(DataPoint<d_t,i_t,v_t> & dp, unsigned int ntrees,
				       int nthreads)
{
  if (ntrees<=0 || ntrees>_dtree_vec.size()) ntrees=_dtree_vec.size();
  MapReduceRunner runner(nthreads,MapReduceRunner::BLOCK);
  class TreeApplyMR : public MapReduce {
  public:
    
    bool is_sorted;

    
    vector<double> scr_arr;

    
    double result;

    
    DecisionTree<d_t,i_t,v_t> *tree_arr;

    
    int ntrees;

    
    DataPoint<d_t,i_t,v_t> * dp_ptr;

    
    void set(DecisionTree<d_t,i_t,v_t> *_tree_arr, int _ntrees, DataPoint<d_t,i_t,v_t> * _dp_ptr, int _nthrds)
    {
      tree_arr = _tree_arr;
      ntrees=_ntrees;
      dp_ptr=_dp_ptr;
      is_sorted=_dp_ptr->is_sorted();
      if (_nthrds>=1) {
	scr_arr.resize(_nthrds);
	for (int i=0; i<_nthrds; i++) scr_arr[i]=0;
      }
      result=0;
    }
  
    void map(int tid, int j) {
      scr_arr[tid] += tree_arr[j].apply(std::ref(*dp_ptr), is_sorted);
    }

    void reduce(int tid) {
      result += scr_arr[tid];
    }
  } mr;
  mr.set(_dtree_vec.data(), ntrees, &dp, runner.nthreads);

  runner.run(mr,0,ntrees);

  return mr.result;
}

template<typename d_t, typename i_t, typename v_t>
size_t DecisionForest<d_t,i_t,v_t>::appendFeatures
(DataPoint<d_t,i_t,v_t> & dp, vector<int> & feat_vec, size_t offset)
{
  bool is_sorted= dp.is_sorted();
  for (int i=0; i<_dtree_vec.size(); i++) {
    offset=_dtree_vec[i].appendFeatures(dp,feat_vec,offset,is_sorted);
  }
  return offset;
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForest<d_t,i_t,v_t>::write(ostream & os)
{
  int i;
  char c='\n';
  MyIO::write<double>(os,this->threshold);
  MyIO::write<int>(os,_dim_dense);
  MyIO::write<int>(os,_dim_sparse);
  MyIO::write<int>(os,_train_loss);
  MyIO::write<int>(os,_dtree_vec.size());
  os.put(c);
  for (i=0; i<_dtree_vec.size();i++) {
    _dtree_vec[i].write(os);
    os.put(c);
  }
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForest<d_t,i_t,v_t>::read(istream & is)
{
  int i,nn;
  char c;
  MyIO::read<double>(is,this->threshold);
  MyIO::read<int>(is,_dim_dense);
  MyIO::read<int>(is,_dim_sparse);
  MyIO::read<int>(is,_train_loss);
  MyIO::read<int>(is,nn);
  is.get(c);
  assert(c=='\n');
  _dtree_vec.resize(nn);
  for (i=0; i<_dtree_vec.size();i++) {
    _dtree_vec[i].read(is);
    is.get(c);
    assert(c=='\n');
  }
}

#include "forest_trainer.h"

namespace rgf {
  template<>
  void DecisionForestFlt::train(DataSetFlt & ds, double* scr_arr, 
				class DecisionTreeFlt::TrainParam &param_dt,
				class DecisionForestFlt::TrainParam & param_rgf,
				DataSetFlt & tst,
				string model_file,
				DataDiscretizationInt *disc_ptr)				
  {
    return;
  }
}


template<typename d_t, typename i_t, typename v_t>
void DecisionForest<d_t,i_t,v_t>::train(
				     DataSet<d_t,i_t,v_t> & ds, double* scr_arr, 
				     class DecisionTree<d_t,i_t,v_t>::TrainParam &param_dt,
				     class DecisionForest<d_t,i_t,v_t>::TrainParam & param_rgf,
				     DataSet<d_t,i_t,v_t> & tst,
				     string model_file,
				     DataDiscretizationInt *disc_ptr)
{
  double *my_scr_arr=scr_arr;

  if (ds.size()<=1) return;

  int ngrps= MapReduceRunner::num_threads(param_dt.nthreads.value);

  if (scr_arr==0) {
    my_scr_arr= new double [ds.size()];
    memset(my_scr_arr,0,sizeof(double)*ds.size());
  }
  _dim_dense=ds.dim_dense();
  _dim_sparse=ds.dim_sparse();
  _train_loss=TrainLoss::str2loss(param_dt.loss.value);
  _dtree_vec.resize(param_rgf.ntrees.value);

  DecisionForestTrainer forest_trainer(param_rgf.opt.value);

  if (param_rgf.verbose.value>=2) {
    cerr << "  training data size= " << ds.size() << " with " << ds.dim_dense() << " dense features"
	 << " and " << ds.dim_sparse() << " sparse feature groups" <<endl;
    fflush(stderr);
  }
  t0.start();
  forest_trainer.init(ds,ngrps,param_rgf.verbose.value);
  t0.stop();
  int eval_freq = tst.size()<=0? 0: param_rgf.eval_frequency.value;
  int write_freq = model_file.size()<=0? 0: param_rgf.write_frequency.value;
  if (tst.size() != tst.y.size()) eval_freq=0;

  if (param_rgf.verbose.value>=2) {
      fprintf(stderr,"\n\nbuild tree            ");
  }

  MapReduceRunner runner(param_dt.nthreads.value,MapReduceRunner::BLOCK);
  class TrainEvalMR : public MapReduce {
  public:
    
    UniqueArray<double> tst_scr;
    
    UniqueArray<int> tst_leaf_node;
    
    
    DataSet<d_t,i_t,v_t> * tst_ptr;

    
    DecisionTree<d_t,i_t,v_t> *tree_ptr;

    
    vector <DecisionTree<d_t,i_t,v_t> >*tree_vec_ptr;
    
    int cur_tree_id=-1;

    
    void map(int tid, int j) {
      DataPoint<d_t,i_t,v_t> dp= (*tst_ptr)[j];
      if (cur_tree_id<0) { 
	tst_scr[j] += tree_ptr->apply(dp,true);
      }
      else { 
	tst_leaf_node[cur_tree_id+j*tree_vec_ptr->size()]=tree_ptr->leaf_node_index(dp,true);
      }
      return;
    }
    
    
    void map_range(int tid, int b, int e) {
      if (cur_tree_id<0) return;
      for (int i=b; i<e; i++) {
	double result=0;
	int *offset=&tst_leaf_node[i*tree_vec_ptr->size()];
	for (int c=0; c<=cur_tree_id; c++) {
	  if (offset[c]>=0) {
	    result += ((*tree_vec_ptr)[c])[offset[c]].prediction;
	  }
	}
	tst_scr[i]=result;
      }
      return;
    }
    
  } mr;

  if (eval_freq>0) {
    tst.sort();

    mr.tst_ptr=&tst;
    mr.tst_scr.reset(tst.size());
    if (forest_trainer.is_fully_corrective()) {
      mr.tst_leaf_node.reset(tst.size()*_dtree_vec.size());
    }
    memset(mr.tst_scr.get(),0,sizeof(double)*tst.size());
  }
  
  for (int i=0; i<_dtree_vec.size();i++) {
    if (param_rgf.verbose.value>=2) {
      fprintf(stderr,"\b\b\b\b\b\b\b\b\b\b\b");
      fprintf(stderr,"%5d/%5d",i+1,(unsigned int)_dtree_vec.size());
      fflush(stderr);
    }
    t1.start();
    forest_trainer.build_single_tree(ds, my_scr_arr, param_dt, param_rgf.step_size.value, _dtree_vec[i]);
    t1.stop();
    if (forest_trainer.is_fully_corrective()) {
      t2.start();
      forest_trainer.fully_corrective_update(ds, my_scr_arr, param_dt, _dtree_vec.data(), i+1);
      t2.stop();
    }
    t3.start();
    if (eval_freq>0) {
      mr.tree_vec_ptr= & _dtree_vec;
      mr.tree_ptr= & _dtree_vec[i];
      if (forest_trainer.is_fully_corrective()) mr.cur_tree_id=i;
      runner.run(mr,0,tst.size());
    }
    if (eval_freq>0 && (i+1)%eval_freq==0) {
      cerr << endl << "*** evaluate with " << (i+1) << " trees ***";
      if (forest_trainer.is_fully_corrective()) {
	runner.run_range(mr,0,tst.size());
      }
      int j;
      cerr << endl << "on training data: ";
      BinaryTestStat train_result(ds.y_type,_train_loss);
      for (j=0; j<ds.size(); j++) {
	train_result.update(ds.y[j],my_scr_arr[j],this->classify(my_scr_arr[j]));
      }
      train_result.print(cerr);
      cerr << "on test     data: ";
      BinaryTestStat test_result(tst.y_type, _train_loss);
      for (j=0; j<tst.size(); j++) {
	test_result.update(tst.y[j], mr.tst_scr[j], this->classify(mr.tst_scr[j]));
      }
      test_result.print(cerr);
      cerr <<endl;
      fflush(stderr);
      if (i+1<_dtree_vec.size()&&!(write_freq>0 && (i+1)%write_freq==0)) {
	fprintf(stderr,"build tree            ");
	fflush(stderr);
      }
    }
    if (write_freq>0 && (i+1)%write_freq==0) {
      string fn = model_file + "-" + std::to_string(i+1);
      ofstream os(fn);
      cerr << endl << "*** save intermediate forest model to " << fn << " ***" <<endl;
      if (!os.good()) {
	cerr << "  cannot open output file" <<endl;
      }
      else {
	DecisionForest<d_t,i_t,v_t> forest_out;
	forest_out=*this;
	forest_out._dtree_vec.resize(i+1);
	if (disc_ptr !=nullptr) forest_out.revert_discretization(*disc_ptr);
	os.precision(10);
	forest_out.write(os);
	os.close();
      }
      cerr << endl;
      if (i+1<_dtree_vec.size()) {
	fprintf(stderr,"build tree            ");
	fflush(stderr);
      }
    }
    t3.stop();
  }
  if (param_rgf.verbose.value>=2) {
    fprintf(stderr,"\n");
  }

  forest_trainer.finish(ds,param_rgf.verbose.value);
  
  if (scr_arr==0) delete [] my_scr_arr;

  
  if (param_rgf.verbose.value>=5) {
    t0.print();
    t1.print();
    if (forest_trainer.is_fully_corrective()) t2.print();
    if (eval_freq>0) t3.print();
  }
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForest<d_t,i_t,v_t>::revert_discretization(DataDiscretizationInt & disc)
{
  _dim_dense=disc.disc_dense.size();
  _dim_sparse=disc.disc_sparse.size();
  for (int i=0; i<_dtree_vec.size(); i++) {
    _dtree_vec[i].revert_discretization(disc);
  }
}

template<typename d_t, typename i_t, typename v_t>
void DecisionForest<d_t,i_t,v_t>::print(ostream & os, vector<string> & feature_names,
					bool depth_first)
{
  for (int i=0; i<_dtree_vec.size(); i++) {
    os << "tree[" << i << "]:" <<endl;
    _dtree_vec[i].print(os, _dim_dense, _dim_sparse, 
			feature_names, depth_first);
  }
}

					

namespace rgf {
  template class DecisionForest<DISC_TYPE_T>;
  template class DecisionForest<int,int,int>;
  template class DecisionForest<float,src_index_t,float>;
}
