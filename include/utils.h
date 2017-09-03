/************************************************************************
 *  utils.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _RGF_UTILS_H

#define _RGF_UTILS_H

#include "header.h"


namespace rgf {
  
  class MyIO {
  public:
    
    static const char delim=' ';

    
    template <typename T>
      static void write(ostream & os, T _v)
      {
	os << _v << delim;
      }

    
    template <typename T>
      static void read(istream & is, T & _v)
      {
	is >> _v;
	char c;
	is.get(c);
	assert(c==delim);
      }
    
    static void write(ostream & os, string _v)
    {
      size_t len=_v.length();
      write<size_t>(os,len);
      for (size_t i=0; i<len; i++) {
	os << _v[i];
      }
      os << delim;
    }
    
    static void read(istream & is, string & _v) 
    {
      size_t len;
      read<size_t>(is,len);
      _v.resize(len);
      for (size_t i=0; i<len; i++) {
	is.get(_v[i]);
      }
      char c;
      is.get(c);
      assert(c==delim);
    }
  };
  

class Timer {
  
  clock_t b_cpu;
  
  clock_t e_cpu;

  
  chrono::system_clock::time_point b_wall;
  
  chrono::system_clock::time_point e_wall;
  
 public:
  
  string description;

  
  double duration_cpu;

  
  double duration_wall;

  
  Timer(string desc="") : b_cpu(0), e_cpu(0), 
		       description(desc), duration_cpu(0), duration_wall(0) {}

  
  void  start() {
    b_cpu=clock();
    b_wall=chrono::system_clock::now();
  }

  
  void stop() {
    e_cpu=clock();
    e_wall=chrono::system_clock::now();
    duration_cpu += ((double)(e_cpu-b_cpu))/CLOCKS_PER_SEC;
    duration_wall += chrono::duration<double,ratio<1,1> >(e_wall-b_wall).count();
    b_cpu=e_cpu;
    b_wall=e_wall;
  }

  
  void print(ostream &os=cerr) {
    os << description << ": " << "wall time=" << duration_wall << " seconds; "
       << "cpu time=" << duration_cpu << " seconds." <<endl;
  }
};


  
  class MapReduce {
  public:
    
    void map(int tid, int j) {}

    
    void map_range(int tid, int begin, int end) {}
    
    
    void reduce (int tid) {}

    
    void master() {}
  };

  template<typename T>
  class MapReduceCounter : public MapReduce {
  public:
    vector<T> counts;
    T result;
    void set_nthreads(int nthreads, T init_value) {
      counts.resize(nthreads);
      for (int tid=0; tid<counts.size(); tid++) counts[tid]=init_value;
      result=init_value;
    }
    
    void reduce(int tid) {
      result += counts[tid];
    }
  };

  
  class MapReduceRunner {

    
    vector<thread> _th;
  public:
    
    
    
    
    
    

  public:
    
    enum par_t {
      
      DYNAMIC=0,
      
      INTERLEAVE=1,
      
      BLOCK=2
    } parallel_mode;
    
    
    int nthreads;

    
    MapReduceRunner(int nthrds=0, enum par_t par_mode=INTERLEAVE) 
    {
      set(nthrds,par_mode);
    }

    
    static unsigned int max_nthreads() {
      int result =std::thread::hardware_concurrency();
      return  result<1? 1: result;
    }

    
    static unsigned int num_threads(int nthrds) {
      int result=nthrds;
      int _max_nthreads=max_nthreads();
      if (result<=0 || result> _max_nthreads) result= _max_nthreads;
      return result;
    }
    
    
    void set(int nthrds=0,enum par_t par_mode=INTERLEAVE) {
      nthreads=num_threads(nthrds);

      _th.resize(nthreads);
      parallel_mode=par_mode;
    }

    
    template<class T>
    void single_thread_map_reduce(T & mr, int begin, int end, int tid, int nthreads, bool run_range)
    {
      int j;

      if (run_range) {
	int block_size= 1+ (int)((end-1-begin)/nthreads);
	int my_begin=begin+ tid*block_size;
	int my_end= min(end,begin+ (tid+1)*block_size);
	mr.map_range(tid,my_begin,my_end);
	return;
      }
      
      switch (parallel_mode) {
      case INTERLEAVE:
	for (j=begin+tid; j<end; j+=nthreads) {
	  mr.map(tid,j);
	}
	break;
	
      default:
	{
	  int block_size= 1+ (int)((end-1-begin)/nthreads);
	  int my_begin=begin+ tid*block_size;
	  int my_end= min(end,begin+ (tid+1)*block_size);
	  for (j=my_begin; j<my_end; j++) mr.map(tid,j);
	}
	

      }
    }

    
    
    template<class T>
    void run_threads(T & mr,int begin, int end, bool run_range) {
      int tid;
      if (nthreads<=1) {
	mr.master();
	single_thread_map_reduce<T>(std::ref(mr),begin, end, 0,1,run_range);
	mr.reduce(0);
	return;
      }
      static const bool use_omp=true;

#ifndef USE_OMP
      for (tid=0; tid<nthreads; tid++) {
	_th[tid]= thread(& MapReduceRunner::single_thread_map_reduce<T>, this,
			 std::ref(mr),begin, end, tid, nthreads,run_range);
      }
#else
      omp_set_num_threads(nthreads);
#pragma omp parallel for
      for (tid=0; tid<nthreads; tid++) {
	single_thread_map_reduce<T>(std::ref(mr),begin,end,tid,nthreads,run_range);
      }
#endif
      
      mr.master();
      for (tid=0; tid<nthreads; tid++) {
#ifndef USE_OMP
	_th[tid].join();
#endif
	mr.reduce(tid);
      }
    }

    
    template<class T>
    void run(T & mr,int begin, int end) {
      run_threads(mr,begin,end,false);
    }
    
    
    template<class T>
    void run_range(T & mr,int begin, int end) {
      run_threads(mr,begin,end,true);
    }
    
  };

  
  template<class T>
  class UniqueArray
  {
    
    UniqueArray(const UniqueArray &) = delete; 
    UniqueArray & operator=(const UniqueArray &) = delete ; 
    
    
    size_t _num;
    
    
    unique_ptr<T []>  _data;
      
  public:
    
    UniqueArray() : _num(0), _data(nullptr) {}

    
    UniqueArray(size_t n) : _num(0), _data(nullptr)
    {
      reset(n);
    }

    
    UniqueArray(UniqueArray &&) = default;
    UniqueArray & operator = (UniqueArray &&) = default;
    
    
    size_t size() {return _num;}

    
    T * get() {return _data.get();}

    
    T * begin() {return get();}

    
    T* end() {return get()+size();}
    
    
    void reset(size_t n) {
      _num=n;
      if (n<=0) _data.reset(nullptr);
      else _data.reset(new T [n]);
    }

    
    void resize(size_t n) {
      if (n <= _num) {
	_num=n;
	return;
      }
      T * ptr= new T [n];
      memcpy(ptr,get(),sizeof(T)*_num);
      _num=n;
      _data.reset(ptr);
    }
    
    
    void clear() {
      _num=0;
      _data.reset(nullptr);
    }

    
    T & operator [] (size_t i) {return _data[i];}
    
  };

  
  class ParameterParser {
  public:
    
    class ParamValueBase {
    public:
      
      string default_value;

      
      string description;

      
      string parsed_value;

      
      bool is_valid;

      
      virtual void set_value()=0;

    };
    
  private:
    
    static string to_string(string str) {return str;}
    
    static string to_string(bool value) {return value?"true":"false";}
    
    template<typename T>
      static string to_string(T value) {return std::to_string(value);}

    
    vector<pair<string, ParamValueBase *> > _kv_table;

    
    string _description;
  public:
    
    template<typename T> 
      class ParamValue: public ParamValueBase {
    public:
      
      T value;
      
      T default_value_T;

    
      ParamValue() {}

    
    void insert(string _key, 
		T _default, 
		string _description,
		ParameterParser * pp,
		bool _is_valid=true) {
      value = default_value_T = _default;
      default_value=to_string(_default);
      parsed_value= default_value;
      description=_description;

      pp->init_insert(_key,this);
      is_valid=_is_valid;
    }

    
    virtual void set_value() {
      if (parsed_value != "") {
	stringstream convert(parsed_value);
	convert >> value;
      }
      else {
	value=default_value_T;
      }
      is_valid=true;
    }

      
      void set_value(T v) {
	value=v;
	parsed_value= to_string(v);
	is_valid=true;
      }
  };

  
  void init_insert(string key, ParamValueBase * value) {
    _kv_table.push_back(pair<string,ParamValueBase*>(key,value));
  }

  
  bool parse_and_assign(string token);

  
  void print_parameters(ostream & os, string indent="  ");
    
  
  void print_options(ostream & os, string indent="  ");

  
  void set_description(string descr) {
    _description=descr;
  }
    
  
  void clear() {
    _kv_table.clear();
  }

  };

  
  class ParameterParserGroup {
    
    vector<ParameterParser *> pp_vec;
  public:
    
    vector<string> unparsed_tokens;

    
    void command_line_parse(int_t argc, char *argv[]); 

    
    void config_file_parse(string filename);
    
    
    void add_parser(ParameterParser *pp) {
      pp_vec.push_back(pp);
    }

    
    int_t parse(string token);


    
    void print_options(ostream & os, string indent="  ", int_t line_skips=2);


  };
  
}

#endif



