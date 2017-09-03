/************************************************************************
 *  forest_predict.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/



#include "data.h"
#include "forest.h"

#define PROG_DESC "  Load a dataset and a forest model and output prediction to file."
#include "parser.h"

class TestParam : public DataSetFlt::IOParam {
public:
  ParamValue<int> ntrees;
  ParamValue<string> output_prediction;
  ParamValue<string> output_feature;
  ParamValue<string> print_forest;
  ParamValue<string> feature_names;

  TestParam(string prefix="tst.") :   DataSetFlt::IOParam(prefix) {
    ntrees.insert(prefix+"ntrees",0,"if nonzero, use ntrees to compute prediction",this);
    output_prediction.insert(prefix+"output-prediction","","if nonempty, output predictions to this file",this);
    output_feature.insert(prefix+"output-feature","","if nonempty, output features to this file",this);
    print_forest.insert(prefix+"print-forest","","if nonempty, print forest to this file",this);
    feature_names.insert(prefix+"feature-names","","if nonempty, read feature names from the file in the format:\n     feature-0-name\n     feature-1-name\n     ...\n   feature-names are used to print forest when data are in dense or sparse (not mixed) format\n",this);
  }
} param_tstfile;


ModelParam param_modelfile("model.");

void parser_init()
{
  ppg.add_parser(&param_config);
  ppg.add_parser(&param_set);
  
  param_modelfile.set_description("model-file options:");
  ppg.add_parser(&param_modelfile);

  param_tstfile.set_description("test-data and output options:");
  ppg.add_parser(&param_tstfile);

}

#include "test_output.h"
class MyTestOutput : public TestOutput<float,int,float> {
public:
  void print_forest(DecisionForestFlt & forest)
  {
    if (param_tstfile.print_forest.value.size()>0) {
      vector<string> feature_names;
      if (param_tstfile.feature_names.value.size()>0) {
	ifstream is(param_tstfile.feature_names.value);
	cerr <<"read feature names from <" << param_tstfile.feature_names.value << ">" <<endl;
	if (is.good()) {
	  string token;
	  
	  while (!is.eof()) {
	    getline(is,token);
	    if (token.size() && !is.eof()) feature_names.push_back(token);
	  }
	}
	else {
	  cerr <<"cannot open file <" << param_tstfile.feature_names.value << ">" <<endl;
	}
	cerr << "number of lines = " << feature_names.size() <<endl;
	is.close();
      }
      ofstream os;
      os.open(param_tstfile.print_forest.value);
      if (os.good()) {
	cerr <<"print trees to <" << param_tstfile.print_forest.value << ">" <<endl;
	forest.print(os,feature_names);
      }
      else {
	cerr <<" cannot open file <" << param_tstfile.print_forest.value << ">" <<endl;
      }
      os.close();
    }
  }
} tst_out;

int main(int argc, char *argv[])
{

  parser_init();
  parse_commandline(argc,argv);

  int nthreads=MapReduceRunner::num_threads(param_set.nthreads.value);
  param_tstfile.nthreads.set_value(nthreads);
  assert(param_tstfile.nthreads.value==nthreads);
  if (param_set.verbose.value>=2) {
    cerr << " using up to " << nthreads << " threads" << endl;
  }

  Timer t;
  DecisionForestFlt forest;
  
  
  if (param_modelfile.load_filename.value.size()>0) {
    cerr << endl <<endl;
    cerr << "loading forest model from <" << param_modelfile.load_filename.value
	 << ">" <<endl;
    ifstream is(param_modelfile.load_filename.value);
    if (!is.good()) {
#ifdef USE_OMP
    cerr << " using up to " << nthreads << " openmp threads" << endl;
#else
    cerr << " using up to " << nthreads << " threads" << endl;
#endif
    }
    else {
      forest.read(is);
    }
    is.close();
  }

  tst_out.print_forest(forest);
  tst_out.read_tstfile();
  tst_out.print_outputs(forest,param_tstfile.ntrees.value,nthreads);

  cout << endl;
}
