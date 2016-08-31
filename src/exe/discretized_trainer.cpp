/************************************************************************
 *  discretized_trainer.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/



#include "data.h"
#include "forest.h"

#define PROG_DESC "  Read a 'discretized' dataset in appropriate format from a file.\n  Train a decision forest, and save the trained model.\n  If a discretizer is provided, the trained model may be reverted to take the original undiscretized features."

#include "parser.h"




#ifdef DISC_BIG
using DataPointDisc=DataPointInt;
using DataSetDisc=DataSetInt;
using DecisionTreeDisc=DecisionTreeInt;
using DecisionForestDisc=DecisionForestInt;
#define DISC_TEMPLATE int,int,int
#else
using DataPointDisc=DataPointShort;
using DataSetDisc=DataSetShort;
using DecisionTreeDisc=DecisionTreeShort;
using DecisionForestDisc=DecisionForestShort;
#define DISC_TEMPLATE DISC_TYPE_T
#endif

class TestParam : public DataSetDisc::IOParam {
public:
  ParamValue<string> output_prediction;
  ParamValue<string> output_feature;
  TestParam(string prefix="tst.") :   DataSetDisc::IOParam(prefix) {
    output_prediction.insert(prefix+"output-prediction","","if nonempty, output predictions to this file",this);
    output_feature.insert(prefix+"output-feature","","if nonempty, output features to this file",this);
  }
};

class DiscModelParam : public ModelParam {
public:
  ParamValue<string> disc_model_file;
  DiscModelParam(string prefix="model.") : ModelParam(prefix) {   
    disc_model_file.insert
      (prefix+"disc_model-file","",
       "if nonempty, load discretization model from this file and revert discretization in the trained trees",this);
  }
};



DataSetDisc::IOParam param_trnfile("trn.");
DiscModelParam param_modelfile("model.");
DecisionForestDisc::TrainParam param_rgf("forest.");
DecisionTreeDisc::TrainParam param_dt("dtree.");
TestParam param_tstfile("tst.");


void parser_init()
{
  ppg.add_parser(&param_config);
  ppg.add_parser(&param_set);
  
  param_trnfile.set_description("training-data options:");
  ppg.add_parser(&param_trnfile);

  param_rgf.set_description("forest training options:");
  ppg.add_parser(&param_rgf);

  param_dt.set_description("decision tree training options:");
  ppg.add_parser(&param_dt);

  param_modelfile.set_description("model-file options:");
  ppg.add_parser(&param_modelfile);

  param_tstfile.set_description("test-data and output options:");
  ppg.add_parser(&param_tstfile);

}

#include "test_output.h"

TestOutput<DISC_TYPE_T> tst_out;


int main(int argc, char *argv[])
{
  Timer t;
  parser_init();
  parse_commandline(argc,argv);
  DecisionForestDisc forest;

  int nthreads=MapReduceRunner::num_threads(param_set.nthreads.value);
  param_trnfile.nthreads.set_value(nthreads);
  param_tstfile.nthreads.set_value(nthreads);
  param_dt.nthreads.set_value(nthreads);
  assert(param_trnfile.nthreads.value==nthreads);
  assert(param_tstfile.nthreads.value==nthreads);
  assert(param_dt.nthreads.value==nthreads);
  if (param_set.verbose.value>=2) {
    cerr << " using up to " << nthreads << " threads" << endl;
  }
  param_rgf.verbose.set_value(param_set.verbose.value);

  bool pre_load= (param_rgf.eval_frequency.value>0)
    && (param_rgf.eval_frequency.value < param_rgf.ntrees.value)
    && param_tstfile.fn_x.value.size()>0;
    
  
  if (param_trnfile.fn_x.value.size()>0) {
    DataSetDisc trn;
    cerr << "loading training data ... " <<endl;
    param_trnfile.print_parameters(cerr);
    t=Timer("loading time");
    t.start();
    trn.append(param_trnfile);
    t.stop();
    t.print();

    if (pre_load) {
      tst_out.read_tstfile();
    }
    cerr <<endl<<endl;
    cout << "training decision forest ... " <<endl;
    param_dt.print_parameters(cerr);
    param_rgf.print_parameters(cerr);

    cerr <<endl<<endl;
    t=Timer("training time");
    t.start();
    forest.train(trn, 0, param_dt,param_rgf,tst_out.tst);
    t.stop();
    t.print();
  }
  
  
  if (param_modelfile.load_filename.value.size()>0) {
    cerr << endl <<endl;
    cerr << "loading forest model from <" << param_modelfile.load_filename.value
	 << ">" <<endl;
    ifstream is(param_modelfile.load_filename.value);
    if (!is.good()) {
      cerr << "cannot open model file for reading " <<endl;
    }
    else {
      forest.read(is);
    }
    is.close();
  }

  
  if (tst_out.tst.size()==0) {
    tst_out.read_tstfile();
  }  
  tst_out.print_outputs(forest,0,nthreads);

  
  if (param_modelfile.disc_model_file.value.size()>0) {
    cerr << "loading data discretization model from <" << param_modelfile.disc_model_file.value
	 << ">" <<endl;
    ifstream is(param_modelfile.disc_model_file.value);
    if (!is.good()) {
      cerr << "cannot open <" << param_modelfile.disc_model_file.value <<">" <<endl;
    }
    DataDiscretizationInt disc;
    disc.read(is);
    is.close();
    cerr << "reverting data discretization in decision forest" <<endl;
    forest.revert_discretization(disc);
  }
  
  
  if (param_modelfile.save_filename.value.size()>0) {
    cerr << endl <<endl;
    cerr << "saving forest model to <" << param_modelfile.save_filename.value
	 << ">" <<endl;
    ofstream os(param_modelfile.save_filename.value);
    if (!os.good()) {
      cerr << "cannot open model file for writing " <<endl;
    }
    else {
      os.precision(10);
      forest.write(os);
    }
    os.close();
  }


}
