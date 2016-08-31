/************************************************************************
 *  test_output.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


template<typename d_t, typename i_t, typename v_t>
class TestOutput {
public:

  DataSet<d_t,i_t,v_t> tst;
  
  void read_tstfile()
  {
    Timer t("loading time");
    if (param_tstfile.fn_x.value.size()>0) {
      cerr << endl <<endl;
      cerr << "loading test data ... " <<endl;
      param_tstfile.print_parameters(cerr);
      t.start();
      tst.append(param_tstfile);
      t.stop();
      t.print();
    }
  }
  

  void print_outputs(DecisionForest<d_t,i_t,v_t> & forest, int ntrees,int nthreads=0)
  {
    if (tst.size()==0) return;
    
    cerr <<endl <<endl;
    cerr << "testing decision forest ... " <<endl;
    BinaryTestStat test_result(tst.y_type, forest.train_loss());

    ofstream os;
    size_t i;
    if (param_tstfile.output_prediction.value.size()>0) {
      cerr <<"output predictions to <" << param_tstfile.output_prediction.value << ">" <<endl;
      os.open(param_tstfile.output_prediction.value);
      if (!os.good()) {
	cerr <<"cannot open <" << param_tstfile.output_prediction.value << ">" <<endl;
      }
    }
    if (os.good()) {
      bool compute_stat= (tst.y.size()==tst.x_dense.size());
      forest.set(param_tstfile.nthreads.value,ntrees);
      if (ntrees>0) cerr << "using "<< ntrees << " trees" << endl;
      for (i=0; i<tst.size(); i++) {
	DataPoint<d_t,i_t,v_t> dp=tst[i];
	double scr=forest.apply(dp,ntrees,nthreads);
	bool pred=forest.classify(scr);
	if (os.good()) os << scr <<endl;
	
	if (compute_stat) {
	  test_result.update(tst.y[i],scr,pred);
	}
      }
      if (param_tstfile.output_prediction.value.size()>0) {
	os.close();
      }
      if (compute_stat) test_result.print(cout);
    }
    
    if (param_tstfile.output_feature.value.size()>0) {
      cerr <<"output features to <" << param_tstfile.output_feature.value << ">" <<endl;
      os.open(param_tstfile.output_feature.value);
      if (!os.good()) {
	cerr <<"cannot open <" << param_tstfile.output_feature.value << ">" <<endl;
	return;
      }

      class FeatWriterMR : public DataSetWriterMapReduce<d_t,i_t,v_t> {
      public:
	DecisionForest<d_t,i_t,v_t> * forest_ptr;
	virtual void write_datapoint(ostream & os, DataSet<d_t,i_t,v_t> & ds, size_t i) {
	  vector<int> feat_vec;
	  DataPoint<d_t,i_t,v_t> dp=ds[i];
	  forest_ptr->appendFeatures(dp,feat_vec,0);
	  for (int j=0; j<feat_vec.size(); j++) os << feat_vec[j] << " ";
	}
      } mr;
      mr.forest_ptr=&forest;
      
      mr.write(os,tst,nthreads);
      os.close();
    }
  }
};
