/************************************************************************
 *  discretized_gendata.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "data.h"

#include "discretization.h"

#define PROG_DESC "  Read a dataset in appropriate format from a file.\n  Train or load a discretizer to discretize its features, and save the 'discretized' dataset to a file.\n  The discretized dataset is used for decision forest training."

#include "parser.h"


#define DISC_TEMPLATE int,int,int
using DataPointDisc=DataPointInt;
using DataSetDisc=DataSetInt;
using FeatureDiscretizationSparseDisc=FeatureDiscretizationSparseInt;
using DataDiscretizationDisc=DataDiscretizationInt;



class OutputParam : public DataSetDisc::IOParam {
public:
  ParamValue<string> convert_type;
  ParamValue<string> output_feature;
  OutputParam(string prefix="output.") :   DataSetDisc::IOParam(prefix) {
    output_feature.insert(prefix+"binary-feature-file","","if nonempty, output features to this file",this);

    convert_type.insert(prefix + "convert_type", "MIX",
			    "discretized feature groups can be MIX (same feature groups as original), or DENSE (convert all sparse into dense) or SPARSE (one sparse feature group)",
			    this);
  }
};


DataSetFlt::IOParam param_infile("input.");
OutputParam param_outfile("output.");
ModelParam param_modelfile("model.");
FeatureDiscretizationDense::TrainParam param_disc_dense("discretize.dense.");
FeatureDiscretizationSparseDisc::TrainParam param_disc_sparse("discretize.sparse.");


void parser_init()
{
  ppg.add_parser(&param_config);
  ppg.add_parser(&param_set);

  param_infile.set_description("input-data options:");
  ppg.add_parser(&param_infile);

  param_outfile.set_description("output-data options:");
  ppg.add_parser(&param_outfile);

  param_modelfile.set_description("model-file options: (if load is empty, train model using input-data)");
  ppg.add_parser(&param_modelfile);

  param_disc_dense.set_description("dense data discretization training options:");
  ppg.add_parser(&param_disc_dense);
  param_disc_sparse.set_description("sparse data discretization training options:");
  ppg.add_parser(&param_disc_sparse);
}

int main(int argc, char *argv[])
{
  parser_init();
  parse_commandline(argc,argv);

  int nthreads=MapReduceRunner::num_threads(param_set.nthreads.value);
  param_infile.nthreads.set_value(nthreads);
  param_outfile.nthreads.set_value(nthreads);
  assert(param_infile.nthreads.value==nthreads);
  assert(param_outfile.nthreads.value==nthreads);
  if (param_set.verbose.value>=2) {
    cerr << " using up to " << nthreads << " threads" << endl;
  }
  
  DataSetFlt input;
  
  cerr << "loading input data ..." <<endl;
  param_infile.print_parameters(cerr);
  input.append(param_infile);
  cerr << endl<<endl;

  DataDiscretizationDisc disc;
  if (param_modelfile.load_filename.value.size()==0) {
    cerr << "training using input data ..." <<endl;
    param_disc_dense.print_parameters(cerr);
    param_disc_sparse.print_parameters(cerr);
    disc.train(input,param_disc_dense,param_disc_sparse,nthreads,param_set.verbose.value);
    disc.set_convert(param_outfile.convert_type.value);
  }
  else {
    cerr << "loading discretization model from <" << param_modelfile.load_filename.value <<">" <<endl;
    ifstream is(param_modelfile.load_filename.value);
    if (!is.good()) {
      cerr << "cannot open <" << param_modelfile.load_filename.value <<">" <<endl;
    }
    disc.read(is);
    is.close();
  }
  if (param_modelfile.save_filename.value.size()>0) {
    cerr << "writing discretization model to <" << param_modelfile.save_filename.value << ">" <<endl;
    ofstream os(param_modelfile.save_filename.value);
    if (!os.good()) {
      cerr << "cannot open <" << param_modelfile.save_filename.value <<">" <<endl;
    }
    os.precision(10);
    disc.write(os);
    os.close();
  }

  DataSetDisc output;
  bool discretized=false;
  if (param_outfile.fn_x.value.size()>0) {
    cerr << "applying discretization to input data" <<endl;
    disc.apply(input,output,nthreads);
    discretized=true;
    cerr << "writing discretized data to <" << param_outfile.fn_x.value <<">"
	 << " with file-format=" << param_outfile.xfile_format.value <<endl;
    output.write(param_outfile);
  }
  if (param_outfile.output_feature.value.size()>0) {
    if (!discretized) {
      cerr << "applying discretization to input data" <<endl;
      disc.apply(input,output,nthreads);
      discretized=true;
    }
    cerr << "writing discretized features to <" << param_outfile.output_feature.value <<">"
	 << endl;

    class DiscFeatWriterMR : public DataSetWriterMapReduce<DISC_TEMPLATE> {
    public:
      
      vector<size_t>offset;
      
      vector<size_t> offset0;
      DataDiscretizationDisc * disc_ptr;
      
      DiscFeatWriterMR(DataDiscretizationDisc & disc) {
	disc_ptr=&disc;
	
	size_t j,k;
	size_t v=0;
	offset.push_back(v);
	for (j=0; j<disc.disc_dense.size(); j++) {
	  offset0.push_back(j);
	  v+=disc.disc_dense[j].size();
	  offset.push_back(v);
	}
	for (j=0; j<disc.disc_sparse.size(); j++) {
	  offset0.push_back(offset.size()-1);
	  for (k=0; k<disc.disc_sparse[j].size(); k++) {
	    v+= disc.disc_sparse[j][k]->size();
	    offset.push_back(v);
	  }
	}
      }
      
      virtual void write_datapoint(ostream & os, DataSetDisc & ds, size_t i)
      {
	DataPointDisc dp=ds[i];
	int j;
	bool first=true;
	for (j=0; j<dp.dim_dense; j++) {
	  int v=dp.x_dense[j];
	  int_t feat=j;
	  int_t sparse_index=0;
	  if (!first) os << " ";
	  first=false;
	  if (v != 0) os << offset[offset0[feat]+sparse_index]+v;
	}
	for (j=0; j<dp.dim_sparse; j++) {
	  for (int k=0; k<dp.x_sparse[j].size(); k++) {
	    if (!first) os << " ";
	    first=false;
	    int_t feat=j+dp.dim_dense;
	    int_t sparse_index=dp.x_sparse[j][k].index;
	    if (sparse_index<0) sparse_index=0;
	    int v=(int)dp.x_sparse[j][k].value;
	    if (v !=0) os << offset[offset0[feat]+sparse_index]+v;
	  }
	}
      }
    } mr(disc);

    
    ofstream os(param_outfile.output_feature.value);
    mr.write(os,output, nthreads);
    os.close();
  }
  
}
