/************************************************************************
 *  feature_mapper.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _DECISIONTREE_TRAINER_FEATURE_MAPPER_H

#define  _DECISIONTREE_TRAINER_FEATURE_MAPPER_H

#include "dtree.h"
#include "training_target.h"



namespace _decisionTreeTrainer
{
  
  
  template<typename i_t, typename v_t>
  class SparseFeatureElementArray 
  {
  public:
    
    int size;
    
    SparseFeatureElement<i_t,v_t>  *data;
    
    SparseFeatureElementArray(int n=0, SparseFeatureElement<i_t,v_t>  * ptr=nullptr) : size(n), data(ptr) {}

    
    bool value_less_or_equal(int index, int cut) {
      if (size<=0) return (0<=cut);
      int b=0;
      int e=size-1;
      while (b<e) {
	int s=(b+e)/2;
	if (data[s].index<index) b=s+1;
	else e=s;
      }
      if (data[b].index==index)return (data[b].value<=cut);
      return (0 <= cut);
    }
  };
  
   
  class FeatureMapper {
    public:
      
      int dim_dense=0;
      
      int dim_sparse=0;
      
      pair<int,int> * original_to_feature=nullptr;
      
      vector<int> * sparse_to_original=nullptr;
      
      vector<int> dense_to_original;

      
      bool is_valid=false;

      
      int to_original(int feature, int sparse_index) {
	return feature<dim_dense?
		       dense_to_original[feature]:
	  sparse_to_original[feature-dim_dense][sparse_index];
      }

      
      template<typename d_t, typename i_t, typename v_t>
      void init(DataSet<d_t,i_t,v_t> & ds, int ngrps)
      {
	
	if (ds.dim_sparse() !=1 || ds.dim_dense()>0 || ngrps<=0) return;
	is_valid=true;
	vector<train_size_t> counts;
	train_size_t i;
	
	for (i=0; i<ds.size(); i++) {
	  for (int k=0; k<ds.x_sparse[i][0].size(); k++) {
	    int ind=(ds.x_sparse[i][0])[k].index;
	    while (ind>=counts.size()) counts.push_back(0);
	    counts[ind]++;
	  }
	}
	dim_sparse=ngrps;
	original_to_feature = new pair<int,int> [counts.size()];
	sparse_to_original = new vector<int> [ngrps];
	int it=0;
	double num=ds.size()+1e-10;
	for (i=0; i<counts.size(); i++) {
	  if (counts[i] >=0.50*num) { 
	    original_to_feature[i]=pair<int,int>(-1,dense_to_original.size());
	    dense_to_original.push_back(i);
	  }
	  else { 
	    int g=it%ngrps;
	    original_to_feature[i]=pair<int,int>(g,sparse_to_original[g].size());
	    sparse_to_original[g].push_back(i);
	    it++;
	  }
	}
	dim_dense=dense_to_original.size();
	if (it<dim_sparse) dim_sparse=it;
      }
      
      ~FeatureMapper() {
	delete [] original_to_feature;
	delete [] sparse_to_original;
      }
  }; 

  
  template<typename d_t>
  class FeatureValueMapDense {
  public:
    
    d_t * my_storage;
    
    int size;
    
    d_t * fv_map_ptr;

    
    double get(int cut) {
      assert(cut >=0 && cut <size);
      if (cut < size-1) {
	return ((double)fv_map_ptr[cut]+(double)fv_map_ptr[cut+1])/2;
      }
      return fv_map_ptr[cut]+0.5;
    }
    
    void clear() {
      delete [] my_storage;
      fv_map_ptr=my_storage=nullptr;
      size=0;
    }
    ~FeatureValueMapDense() {
      clear();
    }
    
    FeatureValueMapDense () : my_storage(nullptr), size(0), fv_map_ptr(nullptr) {}
    
    void initFrom(FeatureValueMapDense<d_t> & parent, bool copy_from_parent,
		  d_t * data_start, int data_size, d_t * tmp_arr)
    {

      int_t psize=parent.size;
      if (copy_from_parent || (data_size >=psize/2)) { 
	my_storage=nullptr;
	size=parent.size;
	fv_map_ptr= parent.fv_map_ptr;
	return;
      }

      
      d_t* new_map_ind=tmp_arr;
      d_t* new_map_value=tmp_arr+(psize+1);
      memset(new_map_ind,0,sizeof(d_t)*(psize+1));
      size_t i;
      
      for (i=0; i<data_size; i++) {
	new_map_ind[1+(int)data_start[i]]=1;
      }

      size=0;
      for (i=1; i<=psize; i++) {
	if (new_map_ind[i]!=0) {
	  new_map_value[size++]=parent.fv_map_ptr[i-1];
	}
	new_map_ind[i] += new_map_ind[i-1];
      }
      my_storage = new d_t [size];
      for (i=0; i<size; i++) {
	my_storage[i]=new_map_value[i];
      }
      fv_map_ptr=my_storage;
      
      for (i=0; i<data_size; i++) {
	data_start[i]=new_map_ind[data_start[i]];
      }
      return;
    }

    
    void initFrom(int disc_value_size) {
      size= disc_value_size;
      my_storage= new d_t [size];
      for (int i=0; i<size; i++) my_storage[i]=i;
      fv_map_ptr = my_storage;
    }

    
    
    void compute_gainLS(int & cut, double & gain,
			double & prediction_current,
			double & prediction_left,
			double & prediction_right,
			double min_sample,
			float lamL1, float lamL2,
			size_t data_size, d_t * data_dense_start,
			YW0_struct *yw_data_arr, double prediction0,
			YW_struct *yw_sum_arr)
    {
      train_size_t i;

      memset(yw_sum_arr,0,size*sizeof(YW_struct));
    
      
      
    
      for (i=0; i<data_size; i++) {
	int v=data_dense_start[i];
	YW0_struct yw0=yw_data_arr[i];
	yw_sum_arr[v].add(yw0.y,yw0.w);
      }

      
      for (i=1; i<size; i++) {
	yw_sum_arr[i].add(yw_sum_arr[i-1]);
      }

      double tot_w=yw_sum_arr[size-1].w;
      double tot_y=yw_sum_arr[size-1].y;

      
      double min_node_weights  = min_sample * tot_w/(data_size +1e-10);

      double obj_current= solve_L1_L2(tot_w, prediction0*tot_w+tot_y, lamL1,lamL2,prediction_current);
      prediction_left=prediction_right=prediction_current;

      
      cut=-1;
      gain=0;

      
      for (int my_cut=0; my_cut<size-1; my_cut++) {
	double y_sum_left=yw_sum_arr[my_cut].y;
	double weight_sum_left=yw_sum_arr[my_cut].w;

	double y_sum_right=tot_y-y_sum_left;
	double weight_sum_right=tot_w-weight_sum_left;

	if (weight_sum_left<min_node_weights ||
	    weight_sum_right<min_node_weights) continue;

	double my_prediction_left;
	double my_obj_left= solve_L1_L2(weight_sum_left, prediction0*weight_sum_left+y_sum_left,
					lamL1, lamL2, my_prediction_left);
	double my_prediction_right;
	double my_obj_right= solve_L1_L2(weight_sum_right, prediction0*weight_sum_right+y_sum_right,
					 lamL1, lamL2, my_prediction_right);
	double my_gain = obj_current - (my_obj_left + my_obj_right);
	if (my_gain>gain+1e-6) {
    	  cut=my_cut;
    	  gain=my_gain;
	  prediction_left=my_prediction_left;
	  prediction_right=my_prediction_right;
	}
      }
      return;
    }
  }; 


  
  class FeatureValueMapSparse {
  public:
    
    int nfeats=0;
    
    int * offset=nullptr;

    
    void reset(int n) {
      delete [] offset;
      nfeats=n;
      offset= new int [n+1];
      memset(offset,0,sizeof(int)*(n+1));
    }
    
    ~FeatureValueMapSparse() {
      delete [] offset;
    }

    
    template<class i_t, class v_t>
    void compute_gainLS(int & sparse_index, int & cut, double & gain,
			double & prediction_current,
			double & prediction_left,
			double & prediction_right,
			double min_sample,
			float lamL1, float lamL2,
			size_t data_size, SparseFeatureElementArray<i_t,v_t> * data_sparse_start,
			YW0_struct *yw_data_arr, double prediction0,
			YW_struct *yw_sum_arr)
    {
      if (nfeats<=0) return;
    
      train_size_t i;
    
      memset(yw_sum_arr,0,offset[nfeats]*sizeof(YW_struct));
      
      
      double tot_w=0.0, tot_y=0.0;

      for (i=0; i<data_size; i++) {
	SparseFeatureElementArray<i_t,v_t> arr=data_sparse_start[i];
	YW0_struct yw0=yw_data_arr[i];
	for (int k=0; k<arr.size; k++) {
	  SparseFeatureElement<i_t,v_t> elem=arr.data[k];
	  int ii=offset[elem.index]+elem.value;
	  yw_sum_arr[ii].add(yw0.y,yw0.w);
	}
	tot_y += yw0.y;
	tot_w += yw0.w;
      }
      
      double min_node_weights  = min_sample * tot_w/(data_size +1e-10);
      
      double obj_current= solve_L1_L2(tot_w, prediction0*tot_w+tot_y, lamL1,lamL2,prediction_current);
      prediction_left=prediction_right=prediction_current;

      
      for (i=0; i<nfeats; i++) {
	int k;
	for (k=offset[i+1]-1; k>offset[i]; k--) {
	  yw_sum_arr[k-1].add(yw_sum_arr[k]);
	}
	yw_sum_arr[offset[i]].set(tot_y,tot_w);
      }
      
      gain=0;
      sparse_index=0;
      cut=offset[1]-offset[0]-1;
      for (int my_index=0; my_index <nfeats; my_index++) {
	
	for (int my_cut=0; my_cut<offset[my_index+1]-offset[my_index]-1; my_cut++) {
	  double y_sum_right=yw_sum_arr[my_cut+offset[my_index]+1].y;
	  double weight_sum_right=yw_sum_arr[my_cut+offset[my_index]+1].w;

	  double y_sum_left=tot_y-y_sum_right;
	  double weight_sum_left=tot_w-weight_sum_right;

	  if (weight_sum_left<min_node_weights ||
	      weight_sum_right<min_node_weights) continue;

	  double my_prediction_left;
	  double my_obj_left= solve_L1_L2(weight_sum_left, prediction0*weight_sum_left+y_sum_left,
					  lamL1, lamL2, my_prediction_left);
	  double my_prediction_right;
	  double my_obj_right= solve_L1_L2(weight_sum_right, prediction0*weight_sum_right+y_sum_right,
					   lamL1, lamL2, my_prediction_right);
	  double my_gain = obj_current - (my_obj_left + my_obj_right);
	  if (my_gain>gain+1e-6) {
	    sparse_index=my_index;
	    cut=my_cut;
	    gain=my_gain;
	    prediction_left=my_prediction_left;
	    prediction_right=my_prediction_right;
	  }
	}
      }
      return;
    }
  }; 

}


#endif
