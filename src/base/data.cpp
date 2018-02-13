/************************************************************************
 *  data.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "data.h"


class MyDataInputException : public std::exception {
public:
  
  string error_message;
  
  int line_no;

  MyDataInputException(string m, int l) : error_message(m), line_no(l) {}
};


template<typename d_t, typename i_t, typename v_t>
class MyDataInputLineParseResult {
public:
  
  double w_val;

  
  double y_val;

  
  vector<d_t> feats_dense;
  
  vector<SparseFeatureGroup<i_t,v_t> > feats_sparse;
  
  vector<SparseFeatureElement<i_t,v_t> > sparse_elem_vec;

  
  string line;

  
  
  static void parse_sparse_element(char * token_str, SparseFeatureElement<i_t,v_t> & result, int lno)
  {
    size_t sep=0;
    for (sep=0; token_str[sep] !=0 && token_str[sep] !=':'; sep++);

    if (token_str[sep] ==0) {
      throw MyDataInputException(" : not in the format of index:value", lno);
    }
    
    token_str[sep]=0;
    
    
    long tmp=atol(token_str);
    if (tmp >= numeric_limits<i_t>::max() ||
	tmp< numeric_limits<i_t>::lowest()) {
      throw MyDataInputException(" : index out of range",lno);      
    }
    result.index=tmp;

    
    if (is_same<v_t,float>::value ||is_same<v_t,double>::value) {
      double tmp= atof(token_str + (sep+1));
      if (tmp>= numeric_limits<float>::max()) tmp=numeric_limits<float>::max();
      if (tmp<= numeric_limits<float>::lowest()) tmp=numeric_limits<float>::lowest();
      if (!(tmp == tmp)) tmp=numeric_limits<float>::lowest(); 
      result.value=(float)tmp;
    }
    else {
      tmp=atol(token_str+ (sep+1));
      if (tmp >= numeric_limits<v_t>::max() ||
	  tmp< numeric_limits<v_t>::lowest()) {
      throw MyDataInputException(" : value out of range",lno);      
      }
      result.value=tmp;
    }
    return;
  }

  
  void parse_x(bool sparse_format, int lno) {
    const char *str=line.c_str();
    vector<char> char_arr;
    do {
      const char *begin = str;
      while (! isspace(*str) && *str) str++;
      if (str ==begin) continue;
      
      string token(begin,str);
      try {
	if (! sparse_format && token.find_first_of('|') ==string::npos) {
	  
	  double tmp=stod(token);
	  if (tmp>= numeric_limits<float>::max()) tmp=numeric_limits<float>::max();
	  if (tmp<= numeric_limits<float>::lowest()) tmp=numeric_limits<float>::lowest();
	  if (!(tmp == tmp)) tmp=numeric_limits<float>::lowest(); 
	  feats_dense.push_back((float)tmp);
	  continue;
	}
	
	SparseFeatureElement<i_t,v_t>  result;
	if (sparse_format) {
	  char_arr.resize(token.size()+1);
	  memcpy(char_arr.data(),token.c_str(), token.size()+1);
	  char *token_str=char_arr.data();
	  parse_sparse_element(token_str,result,lno);
	  
	  sparse_elem_vec.push_back(result);
	  continue;
	}
	sparse_elem_vec.clear();
	char_arr.resize(token.size()+1);
	memcpy(char_arr.data(),token.c_str(), token.size()+1);
	char *token_str=char_arr.data();
	size_t pos;
	bool is_end=false;
	while (true) { 
	  for (pos=0; token_str[pos] !=0 && token_str[pos] !='|'; pos++);
	  if (pos==0) break;
	  if (token_str[pos]==0) {
	    is_end=true;
	  }
	  parse_sparse_element(token_str,result,lno);
	  
	  sparse_elem_vec.push_back(result);
	  if (is_end) break;
	  token_str=token_str + (pos+1);
	}
      }
      catch (MyDataInputException & e) {
	e.error_message= "cannot parse token " + token + e.error_message;
	throw (e);
      }
      
      SparseFeatureGroup<i_t,v_t> tmp(sparse_elem_vec.size());
      for (size_t ii=0; ii< tmp.size(); ii++) {
	tmp[ii]=sparse_elem_vec[ii];
      }
      feats_sparse.push_back(std::move(tmp));
    } while (0 != *str++);

    if (sparse_format) {
      
      SparseFeatureGroup<i_t,v_t> tmp(sparse_elem_vec.size());
      for (size_t ii=0; ii< tmp.size(); ii++) {
	tmp[ii]=sparse_elem_vec[ii];
      }
      feats_sparse.push_back(std::move(tmp));
    }
    return;
  }
};


template<typename d_t, typename i_t, typename v_t>
class MyDataInputLineParserMR : public MapReduce
{
  
  istream *is_x_ptr;
  istream *is_y_ptr;
  istream * is_w_ptr;

  
  bool w_format;
  
  bool y_format;
  
  bool sparse_format;

  
  mutex io_mu;
 public:

  
  bool read_x_only;
  
  bool use_uniform_weights;

  
  bool is_eof;

  
  int lines_read;

  
  vector<MyDataInputLineParseResult<d_t,i_t,v_t> > ps;

  
  MyDataInputLineParserMR(istream & is_x, istream & is_y, istream & is_w, bool y_valid, bool w_valid,
			string is_x_format, int batch_size)
    {
      is_x_ptr= & is_x;
      is_y_ptr=(y_valid)? (& is_y): nullptr;
      is_w_ptr=(w_valid)? (& is_w): nullptr;

      w_format=(is_x_format.find('w') !=string::npos);
      y_format = (is_x_format.find('y')!=string::npos);
      sparse_format = (is_x_format.find("sparse")!=std::string::npos);

      read_x_only = !y_format && !y_valid;      
      use_uniform_weights = !w_format && !w_valid;

      is_eof=false;
      lines_read=0;

      ps.resize(batch_size);
    }

  
  bool read_line(int & j)
  {
    lock_guard<mutex> guard(io_mu);

    if (is_eof) {
      return false;
    }

    if (is_x_ptr==nullptr || is_x_ptr->eof()) {
      is_eof=true;
      return false;
    }
    if (! is_x_ptr->good()) {
      throw MyDataInputException(" invalid feature file",0);      
    }

    if (lines_read>=ps.size()) {
      return false;
    }
    
    j=lines_read;


    
    ps[j].w_val=1.0;
    if (w_format) (*is_x_ptr) >> ps[j].w_val;
    if (is_w_ptr !=nullptr) {
      (*is_w_ptr) >> ps[j].w_val;
    }

    
    ps[j].y_val=0;
    if (y_format) (*is_x_ptr) >> ps[j].y_val;
    if (is_y_ptr !=nullptr) {
      (*is_y_ptr) >> ps[j].y_val;
    }

    
    getline(*is_x_ptr,ps[j].line);
    is_eof= is_x_ptr->eof();

    if (is_w_ptr != nullptr && is_w_ptr->eof() != is_eof) {
      throw MyDataInputException
	("number of lines in weight-file does not match that of feature-file", lines_read);
    }

    if (is_y_ptr !=nullptr && is_y_ptr->eof() != is_eof) {
     throw  MyDataInputException
       ("number of lines in label-file does not match that of feature-file", lines_read);
    }

    if (is_eof) {
      return false;
    }

    lines_read++;
    return true;
  }

  
  void map(int tid, int j)
  {
    int jj;
    
    while (read_line(jj)) {
      
      ps[jj].parse_x(sparse_format,jj);
    }
  }

};

template<typename d_t, typename i_t, typename v_t>
int_t DataSet<d_t,i_t,v_t>::read_nextBatch
(istream & is_x, istream & is_y, istream & is_w, bool y_valid, bool w_valid, string is_x_format, size_t batch_size, int nthreads)
{
  if (is_x.eof()) return 0;
  
  MyDataInputLineParserMR<d_t,i_t,v_t>
    line_parser(is_x,is_y,is_w,y_valid,w_valid,
		is_x_format, batch_size);
  MapReduceRunner runner(nthreads,MapReduceRunner::INTERLEAVE);
  runner.run(line_parser,0,runner.nthreads);
  
  size_t nl=line_parser.lines_read;
  
  int i;
  for (i=0; i<nl; i++) {
    if (_dim_dense<0) _dim_dense= line_parser.ps[i].feats_dense.size();
    if (_dim_dense != line_parser.ps[i].feats_dense.size()) {
      throw  MyDataInputException
	("number of dense features is " +
	 to_string(line_parser.ps[i].feats_dense.size())
	 + " but should have been " + to_string(_dim_dense) + "!",
	 i+1);
    }

    if (_dim_sparse<0) _dim_sparse=line_parser.ps[i].feats_sparse.size();
    if (_dim_sparse != line_parser.ps[i].feats_sparse.size()) {
      throw  MyDataInputException
	("number of sparse features is "
	 + to_string(line_parser.ps[i].feats_sparse.size())
	 + " but should have been " + to_string(_dim_sparse) + "!",
	 i+1);
    }

    
    _nrows++;
    
    if (!line_parser.use_uniform_weights) {
      row_weights.push_back(line_parser.ps[i].w_val);
    }
    if (!line_parser.read_x_only) {
      y.push_back(line_parser.ps[i].y_val);
    }

    int j;

    
    d_t * x_d=nullptr;
    if (_dim_dense >0) {
      x_d = new d_t [_dim_dense];
    }
    for (j=0; j<_dim_dense; j++) {
      x_d[j] = line_parser.ps[i].feats_dense[j];
    }
    x_dense.push_back(x_d);

    
    SparseFeatureGroup<i_t,v_t> * x_s=nullptr;
    if (_dim_sparse >0)  {
      x_s = new SparseFeatureGroup<i_t,v_t> [_dim_sparse];
    }
    for (j=0; j<_dim_sparse; j++) {
      x_s[j] = std::move(line_parser.ps[i].feats_sparse[j]);
    }
    x_sparse.push_back(x_s);
  }
  return nl;
}

template<typename d_t, typename i_t, typename v_t>
size_t DataSet<d_t,i_t,v_t>::append(DataSet::IOParam & param)
{
  ifstream is_x(param.fn_x.value);
  ifstream is_w(param.fn_w.value);
  ifstream is_y(param.fn_y.value);

  bool w_valid= (param.fn_w.value.size()>0);
  bool y_valid= (param.fn_y.value.size()>0);

  if (!is_x.good()) {
    cerr << " cannot open feature file <" << param.fn_x.value << ">" << endl;
    return 0;
  }
  if (w_valid && !is_w.good()) {
    cerr << " cannot open weight file <" << param.fn_w.value << ">" << endl;
    return 0;
  }
  if (y_valid && !is_y.good()) {
    cerr << " cannot open target file <" << param.fn_y.value << ">" << endl;
    return 0;
  }

  y_type = Target(param.y_type.value);

  int batch_size=1000;
  int nthreads=param.nthreads.value;
  
  int_t nlines = 0;
  int nl=0;
  int begin_= is_sorted()? size(): 0;
  while (true) {
    try {
      nl=read_nextBatch(is_x, is_y, is_w, y_valid, w_valid,
			param.xfile_format.value,
			batch_size, nthreads);
    }
    catch (MyDataInputException & e) {
      cerr << " --- error when reading <feature-file,label-file,weight-file>= <" 
	   << param.fn_x.value << "," << param.fn_y.value << "," << param.fn_w.value <<">" 
	   << " at line " << (nlines + e.line_no)
	   << endl;
      cerr << e.error_message <<endl;
      break;
    }
    if (nl==0) break;
    nlines += nl;
  }

  int end_=size();
  {
    for(int i=begin_; i<end_; i++) (*this)[i].sort();
  }
  return nlines;
}

template<typename d_t, typename i_t, typename v_t>
class MyDataSetWriterMR : public DataSetWriterMapReduce<d_t,i_t,v_t> {
public:
  bool write_w;
  bool write_y;
  bool sparse_format;
  char delim='|';

  MyDataSetWriterMR(DataSet<d_t,i_t,v_t> & ds, string xfile_format) {
    write_w=(xfile_format.find('w')!=string::npos);
    write_y=(xfile_format.find('y')!=string::npos);
    sparse_format=(xfile_format.find("sparse")!=string::npos);

    if (sparse_format) {
      if(!(ds.dim_sparse()==1 && ds.dim_dense()==0) && !(ds.dim_sparse()<=0)) {
	cerr << "cannot write as sparse format" <<endl;
	sparse_format=false;
      }
    }
    if (sparse_format) delim=' ';
  }

  virtual void write_datapoint(ostream & os, DataSet<d_t,i_t,v_t> & ds, size_t i) {
    if (write_w) {
      if (ds.row_weights.size()==ds.size()) os << ds.row_weights[i] << " ";
      else os << 1 << " ";
    }
    if (write_y) {
      if (ds.y.size()==ds.size()) os << ds.y[i];
      else os << 0;
    }
    int j;
    d_t *cur_dense=ds.x_dense[i];
    for (j=0; j<ds.dim_dense(); j++) {
      if (sparse_format) {
	if (cur_dense[j])
	  os << " " << j << ":" << cur_dense[j];
      }
      else {
	os << " " <<cur_dense[j];
      }
    }
    SparseFeatureGroup<i_t,v_t> * cur_sparse=ds.x_sparse[i];
    size_t offset=sparse_format? ds.dim_dense():0;
    for (j=0; j<ds.dim_sparse();j++) {
      os << " ";
      if (cur_sparse[j].size()==0) os << delim;
      for (size_t k=0; k<cur_sparse[j].size(); k++) {
	SparseFeatureElement<i_t,v_t> elem= (cur_sparse[j])[k];
	os << elem.index << ":";
	if (is_same<v_t,unsigned char>::value || is_same<v_t,char>::value) {
	  long tmp=offset+(int)elem.value;
	  os << tmp << delim;
	}
	else os << elem.value << delim;
      }
    }
  }

};

template<typename d_t, typename i_t, typename v_t>
void DataSet<d_t,i_t,v_t>::write(IOParam &param, int nthreads)
{
  ofstream os(param.fn_x.value);
  if (!os.good()) {
    cerr << " cannot open file <" << param.fn_x.value <<"> for writing" <<endl;
    return;
  }
  MyDataSetWriterMR<d_t,i_t,v_t> mr(*this, param.xfile_format.value);
  mr.write(os,*this, nthreads);
  os.close();
}

template<typename d_t, typename i_t, typename v_t>
void DataSet<d_t,i_t,v_t>::clear()
{
  row_weights.clear();
  y.clear();

  assert(x_dense.size()==size() && x_sparse.size()==size());
  for (size_t i=0; i<size(); i++) {
    delete [] x_dense[i];
    x_dense[i]=nullptr;
    delete [] x_sparse[i];
    x_sparse[i]=nullptr;
  }
  x_dense.clear();
  x_sparse.clear();

  _nrows = 0;
  _dim_dense = -1;
  _dim_sparse = -1;
}



namespace rgf {
  template class DataSet<float,src_index_t,float>;
  template class DataSet<int,int,int>;
  template class DataSet<DISC_TYPE_T>;
}
