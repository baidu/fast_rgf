/************************************************************************
 *  forest_train.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/



#include "data.h"
#include "discretization.h"
#include "forest.h"


#define PROG_DESC "  Read a dataset in appropriate format from a file.\n  Train a decision forest, and save the trained model to a file."

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

using DataDiscretizationDisc=DataDiscretization<src_index_t,DISC_TYPE_T>;

class TestParam : public DataSetDisc::IOParam {
public:
  ParamValue<string> output_prediction;
  ParamValue<string> output_feature;
};

class TestParamOrig : public DataSetFlt::IOParam {
public:
  ParamValue<string> output_prediction;
  ParamValue<string> output_feature;
  TestParamOrig(string prefix="tst.") :   DataSetFlt::IOParam(prefix) {
    output_prediction.insert(prefix+"output-prediction","","if nonempty, output predictions to this file",this);
    output_feature.insert(prefix+"output-feature","","if nonempty, output features to this file",this);
  }
};



DataSetFlt::IOParam param_trnfile("trn.");
ModelParam param_modelfile("model.");
DecisionForestDisc::TrainParam param_rgf("forest.");
DecisionTreeDisc::TrainParam param_dt("dtree.");
TestParam param_tstfile;
TestParamOrig param_tstfile_orig("tst.");

FeatureDiscretizationDense::TrainParam param_disc_dense("discretize.dense.");
FeatureDiscretizationSparseInt::TrainParam param_disc_sparse("discretize.sparse.");


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

  param_tstfile_orig.set_description("test-data and output options:");
  ppg.add_parser(&param_tstfile_orig);

  param_disc_dense.set_description("dense data discretization training options:");
  ppg.add_parser(&param_disc_dense);
  param_disc_sparse.set_description("sparse data discretization training options:");
  ppg.add_parser(&param_disc_sparse);

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
#ifdef USE_OMP
    cerr << " using up to " << nthreads << " openmp threads" << endl;
#else
    cerr << " using up to " << nthreads << " threads" << endl;
#endif
  }

  if (param_set.nthreads.value <=0) {
    cerr << " the number of threads is set to " << nthreads << ", which is the maximum number of logical hardware threads including hyperthreads" <<endl;
    cerr << " the optimal number of threads is often the number of physical cores that may be smaller than " << nthreads << endl;
    cerr << " for example, to achieve better performance, you may try to set the number of threads to " << nthreads/2 <<endl <<endl;
  }
  
  param_rgf.verbose.set_value(param_set.verbose.value);

  bool pre_load= (param_rgf.eval_frequency.value>0)
    && (param_rgf.eval_frequency.value < param_rgf.ntrees.value)
    && param_tstfile_orig.fn_x.value.size()>0;
    
  
  DataDiscretizationInt disc1;
  DataDiscretizationDisc disc2;
  
  
  if (param_trnfile.fn_x.value.size()>0) {
    DataSetDisc trn;
    {
      
      DataSetFlt trn_orig;
      cerr << "loading training data ... " <<endl;
      param_trnfile.print_parameters(cerr);
      t=Timer("loading time");
      t.start();
      trn_orig.append(param_trnfile);
      t.stop();
      t.print();

      cerr << "discretizing training data ... " <<endl;
      param_disc_dense.print_parameters(cerr);
      param_disc_sparse.print_parameters(cerr);
      t=Timer("discritizer training time");
      t.start();
      DataDiscretizationInt disc;
      disc.train(trn_orig,param_disc_dense,param_disc_sparse,nthreads,param_set.verbose.value);
      disc.set_convert("SPARSE");

      stringstream ss1,ss2;
      disc.write(ss1);
      disc1.read(ss1);

      disc.write(ss2);
      disc2.read(ss2);
      disc2.apply(trn_orig,trn,nthreads);
      t.stop();
      t.print();
    }

    if (pre_load) {
      Timer t("loading time");
      if (param_tstfile_orig.fn_x.value.size()>0) {
	cerr << endl <<endl;
	cerr << "loading test data ... " <<endl;
	param_tstfile_orig.print_parameters(cerr);
	t.start();
	DataSetFlt tst_orig;
	tst_orig.append(param_tstfile_orig);
	cerr << "discretizing test data ..." <<endl;
	disc2.apply(tst_orig,tst_out.tst,param_tstfile.nthreads.value);
	t.stop();
	t.print();
      }
    }
    cerr <<endl<<endl;
    cout << "training decision forest ... " <<endl;
    param_dt.print_parameters(cerr);
    param_rgf.print_parameters(cerr);

    cerr <<endl<<endl;
    t=Timer("training time");
    t.start();
    forest.train(trn, 0, param_dt,param_rgf,tst_out.tst,param_modelfile.save_filename.value,&disc1);
    t.stop();
    t.print();
  }
  
  
  if (tst_out.tst.size()==0) {
    tst_out.read_tstfile();
  }
  param_tstfile.output_prediction.set_value(param_tstfile.output_prediction.value);
  param_tstfile.output_feature.set_value(param_tstfile.output_feature.value);
  tst_out.print_outputs(forest,0,nthreads);

  forest.revert_discretization(disc1);
  
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

  cout << endl;

}
