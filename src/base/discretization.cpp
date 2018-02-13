/************************************************************************
 *  discretization.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "discretization.h"



namespace _discretizationTrainerDense 
{
  
  struct Elem {
    
    float x=0;
    
    float y=0;
    
    float w=0;
    
    bool operator <(const Elem &b) const
    {
      return x < b.x;
    }
    Elem (float _x=0, float _y=0, float _w=0) : x(_x), y(_y), w(_w) {}
  };

  class Bucket {
  public:
    size_t begin;
    size_t end;
    size_t cut;
  
    double gain;

    
    Bucket(size_t _b, size_t _e, Elem * s_arr, 
	   double *y_sum_arr, double * w_sum_arr,
	   double min_bucket_weights,
	   float lamL2) :
      begin(_b), end(_e) , cut(_e), gain(0.0)
    {
      if (min_bucket_weights<1e-3) min_bucket_weights=1e-3;
      if (lamL2<1e-10) lamL2=1e-10;
    
      for (size_t my_cut = begin; my_cut < end; my_cut++) {
	if (s_arr[my_cut].x >=s_arr[my_cut+1].x) {
	  assert(s_arr[my_cut].x==s_arr[my_cut+1].x);
	  continue;
	}
	double y_sum_left=y_sum_arr[my_cut+1]-y_sum_arr[begin];
	double weight_sum_left=w_sum_arr[my_cut+1]-w_sum_arr[begin];
      
	double y_sum_right=y_sum_arr[end+1]-y_sum_arr[my_cut+1];
	double weight_sum_right=w_sum_arr[end+1]-w_sum_arr[my_cut+1];

	if (weight_sum_left+1e-10<min_bucket_weights || 
	    weight_sum_right+1e-10<min_bucket_weights) continue;
	
	double pred_left= y_sum_left/(weight_sum_left+lamL2);
	double pred_right= y_sum_right/(weight_sum_right+lamL2);
	double pred_tot= (y_sum_left+y_sum_right)/(weight_sum_left+weight_sum_right+2*lamL2);
	

	double obj_left=(weight_sum_left+lamL2)*pred_left*pred_left - 2* pred_left*y_sum_left;
	double obj_right=(weight_sum_right+lamL2)*pred_right*pred_right - 2* pred_right*y_sum_right;
	double obj_tot=(weight_sum_left+weight_sum_right+2*lamL2)*pred_tot*pred_tot - 2* pred_tot*(y_sum_left+y_sum_right);
	double my_gain= obj_tot - (obj_left+obj_right);

	if (my_gain>gain) {
	  cut=my_cut;
	  gain=my_gain;
	}
      }
    }

    const bool operator< (const Bucket & b) const {
      return (gain < b.gain);
    }

  };

  float train (UniqueArray<float>& boundaries,
	      double min_bucket_weights,
	       unsigned int  max_buckets, float lamL2,
	       Elem *s, size_t n);
}


float _discretizationTrainerDense::train
(UniqueArray<float> & boundaries,
 double min_bucket_weights,
 unsigned int  max_buckets, float lamL2,
 Elem * s, size_t n)
{
  if (min_bucket_weights<1e-3) min_bucket_weights=1e-3;

  sort(s, s+n);
  
  vector<double> y_sum_vec;
  vector<double> w_sum_vec;
  double y_sum=0;
  double w_sum=0;
  y_sum_vec.push_back(0);
  w_sum_vec.push_back(0);
  size_t i;

  for (i=0; i<n; i++) {
    double w=s[i].w;
    y_sum += s[i].y*w;
    y_sum_vec.push_back(y_sum);
    w_sum += w;
    w_sum_vec.push_back(w_sum);
  }

  double tot_gain=0;
  
  priority_queue<Bucket,vector<Bucket> > qu;
  qu.push(Bucket(0,n-1, s,
		 y_sum_vec.data(), w_sum_vec.data(),
		 min_bucket_weights,lamL2));
  int nbuckets=1;

  vector<float> b_vec;
  
  while (nbuckets <max_buckets && qu.size()>0) {
    Bucket b= qu.top();
    qu.pop();
    if (b.cut>=n-2 || b.gain <=0 || (s[b.cut].x>=s[b.cut+1].x)) continue;
    tot_gain+=b.gain;
    b_vec.push_back(0.5*(s[b.cut].x+s[b.cut+1].x));
    nbuckets++;
    qu.push(Bucket(b.begin,b.cut, s,
		   y_sum_vec.data(), w_sum_vec.data(),
		   min_bucket_weights,lamL2));
    qu.push(Bucket(b.cut+1,b.end, s,
		   y_sum_vec.data(), w_sum_vec.data(),
		   min_bucket_weights,lamL2));
  }
  sort(b_vec.begin(), b_vec.end());
  boundaries.reset(b_vec.size());
  for (i=0; i<b_vec.size(); i++) {
    boundaries[i]=b_vec[i];
  }
  return tot_gain;
}



template<typename i_t>
void FeatureDiscretizationDense::train(DataSet<float,i_t,float> & ds, int j,TrainParam & tr)
{
  using namespace _discretizationTrainerDense;
  UniqueArray<Elem> s;
  s.reset(ds.size());
  double tot_w=1e-10;
  for (size_t i=0; i< ds.size(); i++) {
    Elem tmp;
    tmp.x=ds.x_dense[i][j];
    tmp.y=ds.y[i];
    tmp.w=ds.row_weights.size()>0?ds.row_weights[i]:1.0;
    s[i]=tmp;
    tot_w+=tmp.w;
  }
  
  tot_w=ds.size()/tot_w;
  if (tot_w<1.0) tot_w=1.0;
  for (size_t i=0; i<s.size(); i++) {
    s[i].w *=tot_w;
  }
  _discretizationTrainerDense::train(boundaries,
				     tr.min_bucket_weights.value, 
				     tr.max_buckets.value, tr.lamL2.value,
				     s.get(), s.size());
}


int FeatureDiscretizationDense::apply(float x)
{
  
  int start=0;
  int end=boundaries.size()-1;
  if (end<0 || x > boundaries[end]) return end+1;
  while (end>start) {
    int ii=(start+end)/2;
    if (x <= boundaries[ii]) end=ii;
    else start=ii+1;
  }
  return start;
}

void FeatureDiscretizationDense::read(istream & is) 
{
  int n;
  MyIO::read<int>(is,n);
  boundaries.reset(n);
  for (int i=0; i<n; i++) {
    MyIO::read<float>(is,boundaries[i]);
  }
}

void FeatureDiscretizationDense::write(ostream & os) 
{
  int n=boundaries.size();
  MyIO::write<int>(os,n);
  for (int i=0; i<n; i++) {
    MyIO::write<double>(os,boundaries[i]);
  }

}






template<typename feat_t, typename id_t, typename disc_t>
void FeatureDiscretizationSparse<feat_t,id_t,disc_t>::train
(DataSet<float,feat_t,float> & ds, int j, TrainParam & tr, int nthreads, int verbose)
{


  bool use_omp=false;
#ifdef USE_OMP
  use_omp=true;
#endif

  
  class DataPartition {
  public:
    
    int nthreads; 
    
    UniqueArray<size_t> data_offset; 
    
    vector<size_t> data_index[256];

    
   bool valid() {
     return (nthreads>1);
   }

    
    unsigned int feat2tid(size_t feat) {
      unsigned char r;
      char * s=(char *) & feat;
      for (int j=0; j<sizeof(size_t); j++) 
	r = r * 97 + s[j];
      return ((unsigned int)r)%(unsigned int) nthreads;
    }

    
    bool loop_init(size_t & i, size_t &k, size_t &pos, size_t tid) {
      i=0;
      k=0;
      pos=0;
      return (pos<data_index[tid].size());
    }

    
    bool loop_next(size_t &i, size_t &k, size_t &pos, size_t tid) {
      if (pos>=data_index[tid].size()) return false;
      size_t nitems=data_index[tid][pos];
      while (data_offset[i+1]<=nitems) i++;
      k=nitems-data_offset[i];
      pos++;
      return true;
    }
  } th2data;
  
  th2data.nthreads= (nthreads<=256)?nthreads:0;
  if (th2data.valid()&&use_omp) th2data.data_offset.resize(ds.size()+1); 
    
  Timer t;
  
  using namespace _discretizationTrainerDense;

  t=Timer(" feature_id counting and filtering");
  t.start();

  size_t max_index=0;
  if (th2data.valid()&&use_omp) {
    size_t i;
    size_t nitems=0;
    for (i=0; i<ds.size(); i++) {
      th2data.data_offset[i]=nitems;
      nitems+=((ds.x_sparse[i])[j]).size();
    }
    th2data.data_offset[ds.size()]=nitems;
    UniqueArray<unsigned char> tid_arr(nitems);

#ifdef USE_OMP	
    omp_set_num_threads(th2data.nthreads);
#endif
#pragma omp parallel for
    for (i=0; i<ds.size(); i++) {
      size_t nitems=th2data.data_offset[i];
      auto tmp = & ((ds.x_sparse[i])[j]);
      for (size_t k=0; k<tmp->size(); k++) {
    	size_t feat=(*tmp)[k].index;
    	if(max_index<feat) max_index=feat;
	tid_arr[nitems++]=th2data.feat2tid(feat);
      }
    }
    for (i=0; i<tid_arr.size(); i++) {
      th2data.data_index[tid_arr[i]].push_back(i);
    }
  }
  else {
    for (size_t i=0; i<ds.size(); i++) {
      auto tmp = & ((ds.x_sparse[i])[j]);
      for (size_t k=0; k<tmp->size(); k++) {
	if(max_index<(*tmp)[k].index) max_index=(*tmp)[k].index;
      }
    }
  }
  size_t id=0;
  vector<size_t> id_counts;
  vector<feat_t> id2feat_vec;
  double min_counts= std::max(1,tr.min_occurrences.value);

  UniqueArray<int32_t> feat2id_count_arr;
  bool use_arr= (max_index< (numeric_limits<int32_t>::max()/2-1));
  if (!use_arr) {
    
    unordered_map<feat_t,size_t> feat2id_count_hash;
    for (size_t i=0; i<ds.size(); i++) {
      auto tmp = & ((ds.x_sparse[i])[j]);
      for (size_t k=0; k<tmp->size(); k++) {
	++feat2id_count_hash[(*tmp)[k].index];
      }
    }

    
    for (auto it=feat2id_count_hash.begin(); it != feat2id_count_hash.end(); it++) {
      if (it->second >=min_counts) {
	id_counts.push_back(it->second);
	id2feat_vec.push_back(it->first);
	feat2id[it->first]=id++;
      }
    }
  }
  else {
    
    feat2id_count_arr.resize(max_index+1);
    memset(feat2id_count_arr.get(),0,sizeof(int32_t)*feat2id_count_arr.size());

    if (th2data.valid()&&use_omp) {
#ifdef USE_OMP	
      auto mapper = [j, &th2data, &feat2id_count_arr, &ds] (int tid) {
	size_t pi, i, k, pos;
	if (th2data.loop_init(i,k,pos,tid)) {
	  pi=i;
	  auto tmp = & ((ds.x_sparse[i])[j]);
	  while (th2data.loop_next(i,k,pos,tid)) {
	    if (pi !=i) tmp = & ((ds.x_sparse[i])[j]);
	    ++feat2id_count_arr[(*tmp)[k].index];
	    pi=i;
	  }
	}
      };
      omp_set_num_threads(th2data.nthreads);
#pragma omp parallel for
      for (int tid=0; tid<th2data.nthreads; tid++) {
	mapper(tid);
      }
#endif
    }
    else {
      
      for (size_t i=0; i<ds.size(); i++) {
	auto tmp = & ((ds.x_sparse[i])[j]);
	for (size_t k=0; k<tmp->size(); k++) {
	  ++feat2id_count_arr[(*tmp)[k].index];
	}
      }
    }
    
    for (int ft=0; ft<feat2id_count_arr.size(); ft++) {
      if (feat2id_count_arr[ft] >=min_counts) {
	id_counts.push_back(feat2id_count_arr[ft]);
	id2feat_vec.push_back(ft);
	feat2id_count_arr[ft]=id;
	feat2id[ft]=id++;
      }
      else feat2id_count_arr[ft]=numeric_limits<int32_t>::max();
    }
  }
  t.stop();
  if (verbose>=5) {
    t.print();
  }

  t=Timer(" sparse feature_id to dense");
  t.start();

  
  id2feat.reset(id_counts.size());

  
  class MyID_struct {
  public:
    double w=0; 
    double y=0;  
    size_t count=0;  
    Elem * s_arr_ptr=0; 
    size_t s_arr_size=0; 
  };


  
  size_t tot_counts=0;
  for (id=0; id<id_counts.size(); id++) {
    tot_counts+= (id_counts[id]+1);
  }
  
  UniqueArray<Elem>  s_arr;
  s_arr.resize(tot_counts);

  
  UniqueArray<MyID_struct> id_arr;
  id_arr.resize(id_counts.size());
  tot_counts=0;
  for (id=0; id<id_counts.size(); id++) {
    int sz=(id_counts[id]+1);
    id_arr[id].s_arr_ptr= s_arr.get()+tot_counts;
    id_arr[id].s_arr_size=sz;
    tot_counts += sz;
  }
  
  size_t n=ds.size();

  double tot_w=0; 
  double tot_y=0; 
  
  if (th2data.valid() && use_arr && use_omp) {
#ifdef USE_OMP
    for (size_t i=0; i<n; i++) {
      auto tmp= & ((ds.x_sparse[i])[j]);
      float ww=(ds.row_weights.size()==n)? ds.row_weights[i]:1.0;
      double yy=ds.y[i]*ww;
      tot_y+=yy;
      tot_w+=ww;
    }

    auto mapper = [j, &th2data, &feat2id_count_arr, & id_arr, &ds] (int tid) {
      size_t pi, i, k, pos;
      if (th2data.loop_init(i,k,pos,tid)) {
	pi=i;
	auto tmp = & ((ds.x_sparse[i])[j]);
	float ww=(ds.row_weights.size()==ds.size())? ds.row_weights[i]:1.0;
	double yy=ds.y[i]*ww;

	while (th2data.loop_next(i,k,pos,tid)) {
    
	  if (pi !=i) {
	    tmp = & ((ds.x_sparse[i])[j]);
	    ww=(ds.row_weights.size()==ds.size())? ds.row_weights[i]:1.0;
	    yy=ds.y[i]*ww;
	  }
	  size_t id=feat2id_count_arr[(*tmp)[k].index];
	  if (!(id==numeric_limits<int32_t>::max())) {
	    id_arr[id].w+=ww;
	    id_arr[id].y+=yy;
	    int cnt=++id_arr[id].count;
	    id_arr[id].s_arr_ptr[cnt]=Elem((*tmp)[k].value,yy,ww);
	  }
	  pi=i;
	}
      }
    };
    omp_set_num_threads(th2data.nthreads);
#pragma omp parallel for
    for (int tid=0; tid<th2data.nthreads; tid++) {
      mapper(tid);
    }
#endif
  }
  else {
    for (size_t i=0; i<n; i++) {
      auto tmp= & ((ds.x_sparse[i])[j]);
      float ww=(ds.row_weights.size()==n)? ds.row_weights[i]:1.0;
      double yy=ds.y[i]*ww;
      tot_y+=yy;
      tot_w+=ww;
      for (size_t k=0; k<tmp->size(); k++) {
	if (use_arr) { 
	  id=feat2id_count_arr[(*tmp)[k].index];
	  if (id==numeric_limits<int32_t>::max()) continue;
	}
	else { 
	  auto it = feat2id.find((*tmp)[k].index);
	  if (it == feat2id.end()) continue;
	  id=it->second;
	}
	id_arr[id].w+=ww;
	id_arr[id].y+=yy;
	int cnt=++id_arr[id].count;
	id_arr[id].s_arr_ptr[cnt]=Elem((*tmp)[k].value,yy,ww);
      }
    }
  }
  
  for (id=0; id<id_counts.size(); id++) {
    assert(id_arr[id].count==id_arr[id].s_arr_size-1);
    const float xx=numeric_limits<float>::lowest(); 
    if (tot_w>id_arr[id].w+1e-5) {
      id_arr[id].s_arr_ptr[0]=Elem(xx,(tot_y-id_arr[id].y)/(tot_w-id_arr[id].w),(tot_w-id_arr[id].w));
    }
    else {
      id_arr[id].s_arr_ptr[0]=Elem(xx,0.0,0.0);
    }
  }

  t.stop();
  if (verbose>=5) {
    t.print();
  }
  
  
  struct GainElement {
    
    float value;
    
    int index;
    
    const bool operator< (const GainElement & b) const {
      return (value > b.value);
    }
  };

  t=Timer(" compute gain");
  t.start();
  
  vector<GainElement> gain;
  gain.resize(id_counts.size());
  UniqueArray<UniqueArray<float> > my_boundaries(id_counts.size());
  if (id_counts.size()>0) {
    MapReduceRunner runner(nthreads,MapReduceRunner::INTERLEAVE);
    class SparseDiscMR : public MapReduce {
    public:
      GainElement * gain_ptr;
      MyID_struct  *id_arr_ptr;
      UniqueArray<float> * my_boundaries_ptr;
      TrainParam * tr_ptr;
      void map(int tid, int j) {
	int id=j;
	GainElement tmp;
	tmp.index=id;
	tmp.value=_discretizationTrainerDense::train
	  (my_boundaries_ptr[id],
	   tr_ptr->min_bucket_weights.value,  tr_ptr->max_buckets.value, tr_ptr->lamL2.value,
	   id_arr_ptr[id].s_arr_ptr, id_arr_ptr[id].s_arr_size);
	gain_ptr[j]=tmp;
      }
    } mr;
    mr.gain_ptr=gain.data();
    mr.id_arr_ptr=id_arr.get();
    mr.my_boundaries_ptr=my_boundaries.get();
    mr.tr_ptr= & tr;
    runner.run(mr,0,id_counts.size());

    t.stop();
    if (verbose >=5) {
      t.print();
    }
    
    
  }
  if (id_counts.size()-1 > tr.max_features.value) {
    sort(gain.begin(),gain.end()); 
  }

  
  size_t nf;
  for (nf=min<size_t>(id_counts.size(),tr.max_features.value); nf >0; nf--) {
    if (gain[nf-1].value>0) break;
  }
  boundary_arr.reset(nf);
  feat2id.clear();
  id2feat.reset(nf);
  
  for (size_t j=0; j<nf; j++) {
    id=gain[j].index;
    boundary_arr[j].set(my_boundaries[id]);
    feat_t feat=id2feat_vec[id];
    id2feat[j]=feat;
    feat2id[feat]=j;
  }
  return;
}

template<typename feat_t, typename id_t, typename disc_t>
void FeatureDiscretizationSparse<feat_t,id_t,disc_t>::apply
(SparseFeatureGroup<feat_t,float> & x, vector<SparseFeatureElement<id_t,disc_t> > & result, bool is_sorted)
{
  result.clear();
  for (size_t i=0; i<x.size(); i++) {
    auto tmp=x[i];
    auto it= feat2id.find(tmp.index);
    if (it==feat2id.end()) continue;
    auto id=it->second;
    int value=boundary_arr[id].apply(tmp.value);
    if (value>0) {
      result.push_back(SparseFeatureElement<id_t,disc_t>(id,value));
    }
  }
  if (is_sorted) sort(result.begin(),result.end());
}

template<typename feat_t, typename id_t, typename disc_t>
UniqueArray<SparseFeatureElement<id_t,disc_t> >
FeatureDiscretizationSparse<feat_t,id_t,disc_t>::apply
(SparseFeatureGroup<feat_t,float> & x, bool is_sorted)
{
  vector<SparseFeatureElement<id_t,disc_t> > result_vec;
  apply(x,result_vec,is_sorted);
  UniqueArray<SparseFeatureElement<id_t,disc_t> > result;
  result.reset(result_vec.size());
  for (size_t i=0; i<result.size(); i++) result[i]=result_vec[i];
  return result;
}

template<typename feat_t, typename id_t, typename disc_t>
void FeatureDiscretizationSparse<feat_t,id_t,disc_t>::read(istream & is)
{
  size_t i, n;
  MyIO::read<size_t>(is,n);
  id2feat.reset(n);
  for (i=0; i<n; i++) {
    int id=i;
    size_t feat;
    MyIO::read<size_t>(is,feat);
    id2feat[id]=feat;
    feat2id[feat]=id;
  }
  assert(feat2id.size()==n);
  boundary_arr.reset(n);
  for (i=0; i<n; i++) {
    boundary_arr[i].read(is);
  }
}

template<typename feat_t, typename id_t, typename disc_t>
void FeatureDiscretizationSparse<feat_t,id_t,disc_t>::write(ostream & os)
{
  size_t i, n;
  n=size();
  assert(id2feat.size()==n && feat2id.size()==n && boundary_arr.size()==n);
  MyIO::write<size_t>(os,n);
  for (i=0; i<n; i++) {
    int id=i;
    int feat=id2feat[id];
    MyIO::write<size_t>(os,feat);
  }
  for (i=0; i<n; i++) {
    boundary_arr[i].write(os);
  }
}
  


template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
void DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::train
(DataSet<float,src_i_t,float> & ds,
 FeatureDiscretizationDense::TrainParam & tr_dense,
 class FeatureDiscretizationSparse<src_i_t,dest_i_t,dest_v_t>::TrainParam & tr_sparse,
 int nthreads, int verbose)
{
  if (tr_dense.max_buckets.value+1>=numeric_limits<dest_d_t>::max()) {
    cerr << "maximum dense discretization bucket size " << tr_dense.max_buckets.value
	 << " is more than what's allowed in the currently supporetd discretization type" <<endl;
    cerr << " please reduce the size or recompile with a dense discretization value type allowing larger value" <<endl;
    exit (-1);
  }
  if (tr_sparse.max_buckets.value+1>=numeric_limits<dest_v_t>::max()) {
    cerr << "maximum sparse discretization bucket size " << tr_sparse.max_buckets.value
	 << " is more than what's allowed in the currently supporetd discretization type" <<endl;
    cerr << " please reduce the size or recompile with a sparse discretization value type allowing larger value" <<endl;
    exit (-1);
  }

  MapReduceRunner runner(nthreads,MapReduceRunner::INTERLEAVE);
  
  int j;
  disc_dense.reset(ds.dim_dense());
  if (ds.dim_dense()>0) {
    class DenseDiscMR : public MapReduce {
    public:
      DataSet<float,src_i_t,float> * ds_ptr;
      FeatureDiscretizationDense * disc_dense_ptr;
      FeatureDiscretizationDense::TrainParam * tr_dense_ptr;

      void map(int tid, int j) {
	disc_dense_ptr[j].train(*ds_ptr,j,*tr_dense_ptr);
      }
    } mr;
    mr.ds_ptr=&ds;
    mr.disc_dense_ptr=disc_dense.get();
    mr.tr_dense_ptr=&tr_dense;
    runner.run(mr,0,ds.dim_dense());
  }
  
  disc_sparse.reset(ds.dim_sparse());
  for (j=0; j<ds.dim_sparse(); j++) {
    disc_sparse[j].train(ds,j,tr_sparse,nthreads,verbose);
  }

  offset_init();
  
  return;
}


template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
void DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::revert(int_t &feat, int_t &sparse_index, double & cut)
{
  size_t index;
  FeatureDiscretizationDense * disc_ptr=nullptr;
  if (convert_type==MIX) {
    if (feat >= disc_dense.size()) { 
      int_t id=sparse_index;
      sparse_index= disc_sparse[feat-disc_dense.size()].id2feat[id];
      disc_ptr= disc_sparse[feat-disc_dense.size()][id];
    }
  }
  else { 
    if (convert_type==DENSE) {
      
      index=feat;
    }
    else {
      
      assert(feat<=0);
      index=sparse_index;
    }

    if (index<disc_dense.size()) { 
      feat=index;
      sparse_index=-1;
      disc_ptr = &disc_dense[feat];
    }
    else { 
      for (feat=disc_sparse.size()-1; feat>=1; feat--) {
	if (index >=_offset[feat]) break;
      }
      size_t id=index-_offset[feat];
      sparse_index=disc_sparse[feat].id2feat[id];
      disc_ptr= disc_sparse[feat][id];
      feat += disc_dense.size();
    }
  }
  
  int v= (int) (cut +1e-10);
  cut = 0.5*((*disc_ptr)[v].second + (*disc_ptr)[v+1].first);
}


template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
bool DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::apply
(DataPoint<float,src_i_t,float> & src,
 DataPoint<dest_d_t,dest_i_t,dest_v_t> & dest,bool is_sorted)
{
  if ((disc_dense.size() != src.dim_dense) ||(disc_sparse.size() != src.dim_sparse)) return false;
  int j;
  if (convert_type==MIX) {
    dest.dim_dense=src.dim_dense;
    if (src.dim_dense >0)  dest.x_dense = new dest_d_t [src.dim_dense];
    else dest.x_dense=nullptr;
    for (j=0; j<src.dim_dense;j++) {
      dest.x_dense[j]=disc_dense[j].apply(src.x_dense[j]);
    }

    dest.dim_sparse=src.dim_sparse;
    if (src.dim_sparse >0)  dest.x_sparse = new SparseFeatureGroup<dest_i_t,dest_v_t> [src.dim_sparse];
    else dest.x_sparse=nullptr;
    for (j=0; j<src.dim_sparse;j++) {
      dest.x_sparse[j]=disc_sparse[j].apply(src.x_sparse[j],is_sorted);
    }
    return true;
  }
  
  vector< SparseFeatureElement<dest_i_t,dest_v_t> > result;
  for (j=0; j<src.dim_dense; j++) {
    int vv=disc_dense[j].apply(src.x_dense[j]);
    int ii=j;
    if (vv !=0) {
      result.push_back( SparseFeatureElement<dest_i_t,dest_v_t>(ii,vv));
    }
  }
  vector< SparseFeatureElement<dest_i_t,dest_v_t> > result_tmp;
  for (j=0; j<src.dim_sparse; j++) {
    disc_sparse[j].apply(src.x_sparse[j],result_tmp,true);
    for (size_t k=0; k<result_tmp.size(); k++) {
      auto tmp=result_tmp[k];
      tmp.index += _offset[j];
      result.push_back(tmp);
    }
    result_tmp.clear();
  }
  
  if (convert_type==DENSE) {
    dest.dim_dense=_offset[_offset.size()-1];
    dest.dim_sparse=0;
    dest.x_sparse=nullptr;
    dest.x_dense = new dest_d_t [dest.dim_dense];
    memset(dest.x_dense,0,sizeof(dest_d_t)*dest.dim_dense);
    for (j=0; j<result.size(); j++) {
      dest.x_dense[result[j].index]=result[j].value;
    }
  }
  
  if (convert_type==SPARSE) {
    dest.dim_dense=0;
    dest.dim_sparse=1;
    dest.x_dense =nullptr;
    dest.x_sparse=new SparseFeatureGroup<dest_i_t,dest_v_t> [1];
    dest.x_sparse[0].reset(result.size());
    for (j=0; j<dest.x_sparse[0].size(); j++) {
      dest.x_sparse[0][j]=result[j];
    }
  }
  return true;
}

template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
void DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::apply
(DataSet<float,src_i_t,float> & src,
 DataSet<dest_d_t,dest_i_t,dest_v_t> & dest, int nthreads)
{
  
  dest.clear();
  dest.y_type=src.y_type;

  if (convert_type==MIX) {
    dest.set_dims(src.dim_dense(),src.dim_sparse());
  }
  if (convert_type==DENSE) {
    dest.set_dims(_offset[_offset.size()-1],0);
  }
  if (convert_type==SPARSE) {
    dest.set_dims(0,1);
  }

  MapReduceRunner runner(nthreads,MapReduceRunner::INTERLEAVE);
  class ApplyMR : public MapReduce {
  public:
    UniqueArray<DataPoint<dest_d_t,dest_i_t,dest_v_t> > dest_arr;
    DataSet<float,src_i_t,float> * src_ptr;
    DataDiscretization<src_i_t, dest_d_t, dest_i_t, dest_v_t> * disc_ptr;
    void map(int tid, int i) {
      DataPoint<float,src_i_t,float> dp_src=(*src_ptr)[i];
      disc_ptr->apply(dp_src,std::ref(dest_arr[i]),true);
    }
  } mr;
  mr.dest_arr.reset(src.size());
  mr.src_ptr=&src;
  mr.disc_ptr=this;
  runner.run(mr,0,src.size());
  
  for (size_t i=0; i<src.size(); i++) {
    double *yptr=0;
    float *wptr=0;
    if (src.row_weights.size()==src.size()) wptr=&src.row_weights[i];
    if (src.y.size()==src.size()) yptr=&src.y[i];
    dest.append(mr.dest_arr[i], yptr, wptr);
  }
}

template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
void DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::read(istream & is)
{
  int j,n;

  MyIO::read<int>(is,n);
  disc_dense.reset(n);
  for (j=0; j<n; j++) {
    disc_dense[j].read(is);
  }

  MyIO::read<int>(is,n);
  disc_sparse.reset(n);
  for (j=0; j<n; j++) {
    disc_sparse[j].read(is);
  }

  int cv;
  MyIO::read<int>(is,cv);
  convert_type=static_cast<convert_t>(cv);
  offset_init();
}

  
template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
void DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::write(ostream & os)
{
  int j, n;

  n=disc_dense.size();
  MyIO::write<int>(os,n);
  for (j=0; j<n; j++) {
    disc_dense[j].write(os);
  }

  n=disc_sparse.size();
  MyIO::write<int>(os,n);
  for (j=0; j<n; j++) {
    disc_sparse[j].write(os);
  }

  int cv=(int) convert_type;
  MyIO::write<int>(os,cv);
  
}

template<typename src_i_t, typename dest_d_t, typename dest_i_t, typename dest_v_t>
void DataDiscretization<src_i_t,dest_d_t,dest_i_t,dest_v_t>::offset_init()
{
  _offset.clear();
  size_t v=disc_dense.size();
  _offset.push_back(v);
  for (int j=0; j<disc_sparse.size(); j++) {
    v+= disc_sparse[j].size();
    _offset.push_back(v);
  }
}


namespace rgf {
  template class DataDiscretization<src_index_t,DISC_TYPE_T>;
  template class DataDiscretization<src_index_t,int,int,int>;
}

