/************************************************************************
 *  dtree.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "dtree.h"

void TreeNode::write(ostream & os)
{
  MyIO::write<int_t>(os,feature);
  MyIO::write<int_t>(os,sparse_index);
  MyIO::write<double>(os,cut);
  MyIO::write<double>(os,prediction);
  MyIO::write<int>(os,left_index);
  MyIO::write<int>(os,right_index);
}

void TreeNode::read(istream & is)
{
  MyIO::read<int_t>(is,feature);
  MyIO::read<int_t>(is,sparse_index);
  MyIO::read<double>(is,cut);
  MyIO::read<double>(is,prediction);
  MyIO::read<int>(is,left_index);
  MyIO::read<int>(is,right_index);
}

template<typename d_t, typename i_t, typename v_t>
size_t DecisionTree<d_t,i_t,v_t>::appendFeatures(DataPoint<d_t,i_t,v_t> & dp,
						 vector<int> & feat_vec, size_t offset, bool is_sorted)
{
  size_t nfeatures = _nodes_vec.size();
  int current_index=_root_index;
  while (current_index>=0) {
    int next_index = _nodes_vec[current_index].nextNodeIndex(dp,is_sorted);
    if (next_index <0) {
      feat_vec.push_back(offset+current_index);
      break;
    }
    current_index=next_index;
  }
  return offset+nfeatures;
}

template<typename d_t, typename i_t, typename v_t>
void DecisionTree<d_t,i_t,v_t>::write(ostream & os)
{
  int i;
  MyIO::write<int>(os,_root_index);
  MyIO::write<int>(os,_nodes_vec.size());
  for (i=0; i< _nodes_vec.size(); i++) {
    _nodes_vec[i].write(os);
  }
}

template<typename d_t, typename i_t, typename v_t>
void DecisionTree<d_t,i_t,v_t>::read(istream & is)
{
  clear();
  int i, nn;

  MyIO::read<int>(is,_root_index);
  MyIO::read<int>(is,nn);
  _nodes_vec.resize(nn);
  for (i=0; i<_nodes_vec.size(); i++) {
    _nodes_vec[i].read(is);
  }
}


#include "forest_trainer.h"

namespace rgf {
  template<>
  void DecisionTreeFlt::train(DataSetFlt & ds, double * scr_arr,
			      class DecisionTreeFlt::TrainParam & param_dt)
  {
    return;
  }
}


template<typename d_t, typename i_t, typename v_t>
void DecisionTree<d_t,i_t,v_t>::train(DataSet<d_t,i_t,v_t> & ds, double * scr_arr,
				      class DecisionTree<d_t,i_t,v_t>::TrainParam & param_dt)
{
  DecisionForestTrainer forest_trainer;
  int ngrps= MapReduceRunner::num_threads(param_dt.nthreads.value);
  forest_trainer.init(ds,ngrps,0);
  forest_trainer.build_single_tree(ds, scr_arr, param_dt,1.0,*this);
  forest_trainer.finish(ds,0);
}

template<typename d_t, typename i_t, typename v_t>
void DecisionTree<d_t,i_t,v_t>::revert_discretization(DataDiscretizationInt & disc)
{
  for (int i=0; i<_nodes_vec.size(); i++) {
    if (! _nodes_vec[i].is_leaf()) {
      disc.revert(_nodes_vec[i].feature, _nodes_vec[i].sparse_index, _nodes_vec[i].cut);
    }
  }
  return;
}
				      
static string my_feats(int dim_dense, int dim_sparse, int_t feat, int_t sparse_feat, vector<string> & feature_names)
{
  int_t v=-1;
  if (dim_dense==0 && dim_sparse==1) v = sparse_feat;
  if (dim_dense>=1 && dim_sparse==0) v= feat;
  if (v>=0) {
    if (v < feature_names.size()) return feature_names[v];
    return std::to_string(v);
  }
  return std::to_string(feat) + "|" + std::to_string(sparse_feat);
}


static void print_node(TreeNode *ptr_start, int index, int level, int cur, int & count,
		       int dim_dense, int dim_sparse, ostream & os,
		       vector<string> & feature_names)
{
  auto *ptr = & ptr_start[index];
  os << " ";
  for (int j=0; j<level; j++) os << "    ";
  os << cur << ":";
  if (ptr->is_leaf()) {
    os << "prediction=" << ptr->prediction <<endl;
  }
  else {
    assert(ptr->left_index>=0 && ptr->right_index>=0);
    int left=count++;
    int right=count++;
    os << "[" << my_feats(dim_dense, dim_sparse, ptr->feature, ptr->sparse_index, feature_names)
       << "<" << (ptr->cut + 1e-10) << "] ";
    os << "yes/missing=" << left << ","
       << "no=" << right <<endl;
    print_node(ptr_start,ptr->left_index, level+1, left, count, dim_dense, dim_sparse,os,feature_names);
    print_node(ptr_start,ptr->right_index, level+1, right, count, dim_dense, dim_sparse,os,feature_names);
  }
}
		       
template<typename d_t, typename i_t, typename v_t>
void DecisionTree<d_t,i_t,v_t>::print(ostream & os, int dim_dense, int dim_sparse, 
				      vector<string> & feature_names, bool depth_first)
{
  if (_nodes_vec.size()<=0) {
    if (depth_first) {
      os <<" 0:prediction=0" <<endl;
    }
    else {
      os <<" 0: leaf=0" <<endl;
    }
    return;
  }

  if (depth_first) {
    int count=1;
    print_node(_nodes_vec.data(),_root_index,0,0, count, dim_dense, dim_sparse, os,feature_names); 
    return;
  }

  vector<int> level;
  level.resize(_nodes_vec.size());
  level[0]=0;

  for (int i=0; i<_nodes_vec.size(); i++) {
    auto *ptr = &_nodes_vec[i];
    for (int j=0; j<level[i]; j++) os << "    ";
    os << i << ": ";
    if (ptr->is_leaf()) {
      os << "leaf=" << ptr->prediction <<endl;
    }
    else {
      assert(ptr->left_index>=0 && ptr->right_index>=0);
      os << "[" << my_feats(dim_dense, dim_sparse, ptr->feature, ptr->sparse_index, feature_names)
	 << "<" << (ptr->cut + 1e-10) << "] ";
      os << "yes=" << ptr->left_index << ","
	 << "no=" << ptr->right_index << ","
	 << "missing=" << ptr->left_index <<endl;
      level[ptr->left_index] = level[i]+1;
      level[ptr->right_index] = level[i]+1; 
    }
  }
}


namespace rgf {
  template class DecisionTree<DISC_TYPE_T>;
  template class DecisionTree<int,int,int>;
  template class DecisionTree<float,src_index_t,float>;
}
