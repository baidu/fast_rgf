/************************************************************************
 *  data.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _RGF_DATA_H

#define _RGF_DATA_H


#include "utils.h"

namespace rgf {

  
  template<typename i_t, typename v_t>
  class SparseFeatureElement{
  public:
    
    i_t index;
    
    v_t value;

    
    SparseFeatureElement(i_t i=0, v_t v=0) : index(i), value(v) {}
	
    
    const bool operator<(const SparseFeatureElement & b) const {
      return index < b.index;
    }
  };

  
  template<typename i_t, typename v_t>
  using SparseFeatureGroup = UniqueArray<SparseFeatureElement<i_t,v_t> > ;

  
  template<typename d_t, typename i_t, typename v_t>
  class DataPoint {
  public:
    
    int dim_dense;
    
    d_t *x_dense;

    
    int dim_sparse;
    
    SparseFeatureGroup<i_t,v_t> *x_sparse;

    
    DataPoint(int d0_d=0, d_t *x0_d=nullptr, int d0_s=0, SparseFeatureGroup<i_t,v_t> *x0_s=nullptr) :
      dim_dense(d0_d), x_dense(x0_d), dim_sparse(d0_s), x_sparse(x0_s) {}

    
    bool is_sorted()
    {
      for (int j=0; j<dim_sparse; j++) {
	for (int k=1; k<x_sparse[j].size();k++) {
	  if (x_sparse[j][k].index <= x_sparse[j][k-1].index) return false;
	}
      }
      return true;
    }

    
    void sort()
    {
      if (is_sorted()) return;
      for (int j=0; j<dim_sparse; j++) {
	std::sort(x_sparse[j].begin(),x_sparse[j].end());
	int kk=0;
	for (int k=1; k<x_sparse[j].size();k++) {
	  if (x_sparse[j][k].index>x_sparse[j][kk].index) {
	    kk++;
	    if (kk<k) x_sparse[j][kk]=x_sparse[j][k];
	  }
	}
	x_sparse[j].resize(kk+1);
      }
      return;
    }
    
  };

  
  class Target {
    
    int_t _num_classes;
    
    int_t _current_label;

  public:
    
    enum type_t {
      NUL=0, 
      REAL=1, 
      BINARY=2, 
      MULTICLASS=3 
    } type;


    
    inline bool binary_label(double y) {
      return abs(y-_current_label)<1e-5;
    }

    
    Target() : _num_classes(-1), _current_label(-1), type(NUL) {}

    
    Target(string str) {
      type= NUL;
      _num_classes=-1;
      _current_label=-1;
      if (str.compare("REAL")==0) {
	type=REAL;
      }
      if (str.compare("BINARY")==0) {
	type=BINARY;
	_num_classes=2;
	_current_label=1;
      }
      if (str.compare("MULTICLASS")==0) {
	type=MULTICLASS;
	_num_classes=-1;
	_current_label=0;
      }
    }
  };
  
  
  template<typename d_t, typename i_t, typename v_t>
  class DataSet {
    
    DataSet(const DataSet &); 
    DataSet & operator=(const DataSet &); 
    
    
    size_t _nrows;
    
    
    int _dim_dense;
    
    int _dim_sparse;

    
    bool sorted;
  public:
    
    class IOParam : public ParameterParser {
    public:
      
      ParamValue<string> y_type;
      
      
      ParamValue<string> xfile_format;

      
      ParamValue<string> fn_x;
      
      ParamValue<string> fn_y;
      
      ParamValue<string> fn_w;

      
      ParamValue<int> nthreads;

      
      IOParam(string prefix="data.")
      {
	y_type.insert(prefix+"target",
		      "BINARY",
		      "target type of REAL or BINARY or MULTICLASS",
		      this);
	xfile_format.insert(prefix+"x-file_format",
			    "x",
			    "format: x y.x w.y.x sparse y.sparse w.y.sparse",
			    this);
	fn_x.insert(prefix+"x-file",
		    "",
		    string("feature file name: file format is one data per line\n") 
		    + "    [w] [y] feature-0 ... feature-d\n    ...\n"
		    + "         w is present if x-file_format contains w.\n"
		    + "         y is present if x-file_format contains y.\n"
		    + "         default feature format:\n" 
		    + "                 either       value                 for dense feature\n"
		    + "                 or      index:value|[index:value|] for sparse feature.\n"
		    + "         if x-file_format contains sparse, then feature format is sparse: index:value.\n"
		    + "     ",
		    this);
	fn_y.insert(prefix+"y-file",
		    "",
		    "label file: one label per line (higher priority than y in feature-file)",
		    this);
	fn_w.insert(prefix+"w-file",
		    "",
		    "data weight file: one weight per line (higher priority than w in feature-file)",
		    this);
	
	
      }

    };
    
    
    Target y_type;

    
    vector<float> row_weights;

    
    vector<double> y;

    
    vector<d_t *> x_dense;
    
    vector<SparseFeatureGroup<i_t,v_t> *> x_sparse;


    
    DataSet() :  _nrows(0), _dim_dense(-1), _dim_sparse(-1), sorted(false) {}

    
    size_t size() {
      return _nrows;
    }

    
    bool is_sorted() {return sorted;}
    
    
    DataPoint<d_t,i_t,v_t> operator [] (const size_t i) {
      return DataPoint<d_t,i_t,v_t>(_dim_dense,x_dense[i],
				    _dim_sparse,x_sparse[i]);
    }

    
    void set_dims(int dense, int sparse) {
      _dim_dense=dense;
      _dim_sparse=sparse;
    }
    
    
    int dim_dense() {
      return _dim_dense;
    }

    
    int dim_sparse() {
      return _dim_sparse;
    }

    
    void sort() {
      if (sorted) return;
      for (size_t i=0; i<size(); i++) (*this)[i].sort();
      sorted=true;
    }
    
    
    int_t read_nextBatch(istream & is_x, istream & is_y, istream & is_w,
			 bool y_valid, bool w_valid, string is_x_format,
			 size_t batch_size, int nthreads);
    
    
    size_t append(IOParam & param);

    
    void append(DataPoint<d_t,i_t,v_t> & dp, double *yptr, float *wptr)
    {
      x_dense.push_back(dp.x_dense);
      dp.x_dense=0;
      x_sparse.push_back(dp.x_sparse);
      dp.x_sparse=0;
      if (yptr !=nullptr) y.push_back(*yptr);
      if (wptr !=nullptr) row_weights.push_back(*wptr);
      _nrows++;
    }
		
    
    void read(IOParam & param) {
      clear();
      append(param);
    }
    
    
    void write(IOParam & param, int nthreads=0);

    
    void clear();
    
    
    ~DataSet() {
      clear();
    }
  };

  
  template<typename d_t, typename i_t, typename v_t>
  class DataSetWriterMapReduce : public MapReduce {
    
    UniqueArray<stringstream> str_s;

    
    int current_batch_size;

    
    size_t begin;

    
    DataSet<d_t,i_t,v_t> * ds_ptr;

  public:
    
    int batch_size=100;

    
    void map(int tid, int j) {
      if (j>=current_batch_size) return;
      size_t i=begin+j;
      write_datapoint(str_s[j],*ds_ptr,i);
      str_s[j] <<endl;
    }

    
    void write(ostream & os, DataSet<d_t,i_t,v_t> & ds, int nthreads=0) {
      MapReduceRunner runner(nthreads);
      current_batch_size=batch_size;
      for (begin=0; begin< ds.size(); begin+=current_batch_size) {
	
	if (current_batch_size> (ds.size()-begin)) current_batch_size=(ds.size()-begin);
	str_s.reset(current_batch_size);
	ds_ptr=&ds;
	runner.run(*this,0,current_batch_size);
	
	for (int j=0; j<current_batch_size; j++) {
	  os.write(str_s[j].str().c_str(),str_s[j].str().size());
	}
      }
    }

    
    virtual void write_datapoint(ostream & os, DataSet<d_t,i_t,v_t> & ds, size_t i) =0;
  };
  
#define DISC_SPARSE_T disc_sparse_index_t,disc_sparse_value_t
#define DISC_TYPE_T   disc_dense_value_t,disc_sparse_index_t,disc_sparse_value_t

  using DataPointFlt=DataPoint<float,src_index_t,float>;
  using DataPointInt=DataPoint<int,int,int>;
  using DataPointShort=DataPoint<DISC_TYPE_T>;

  using DataSetFlt=DataSet<float,src_index_t,float>;
  using DataSetInt=DataSet<int,int,int>;
  using DataSetShort=DataSet<DISC_TYPE_T>;

}



#endif

