/************************************************************************
 *  parser.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "utils.h"


class ModelParam : public ParameterParser {
public:
  ParamValue<string> save_filename;
  ParamValue<string> load_filename;

  ModelParam(string prefix="model.")
  {
    save_filename.insert(prefix+"save","","if nonempty, save trained  model to file",this);
    load_filename.insert(prefix+"load","","if nonempty, load previously trained model from file",this);
  }
};


class ConfigParam : public ParameterParser {
public:
  ParamValue<string> filename;
  
  ConfigParam()
  {
    filename.insert("-config","","if nonempty, read options from config-file",this);
  }
};

class SetParam : public ParameterParser {
public:
  
  ParamValue<int> nthreads;
  
  ParamValue<int> verbose;
  SetParam(string prefix="set.") {
    nthreads.insert(prefix+"nthreads", 0, "number of threads for training and testing (0 means maximum number of hardware logical threads)",this);
    verbose.insert(prefix+ "verbose",2, "verbose level",this);
    this->set_description("global options:");
  }
};

ConfigParam param_config;
SetParam param_set;

ParameterParserGroup ppg;

void usage(int argc, char *argv[])
{
  cerr << argv[0] << " " << VER <<endl;
  cerr << PROG_DESC <<endl;
  cerr << endl << "usage:" << " ";
  cerr << argv[0] << " [options]" <<endl <<endl;
  cerr << " options:" <<endl;
  cerr << "  -h [-help | --help] :" ;
  cerr << "   print this help" <<endl<<endl;
  cerr << "   options can be read from commandline or configuration file" <<endl;
  cerr << "                   (commandline overwrites configuration file)"   << endl;
  ppg.print_options(cerr);
  exit(0);
}

void parse_commandline(int argc, char *argv[])
{
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-h") ||!strcmp(argv[i],"-help")
	||!strcmp(argv[i],"--help")) {
      usage(argc,argv);
    }
  }
  ppg.command_line_parse(argc,argv);
  if (ppg.unparsed_tokens.size()>0) {
    cerr << "unknown option " << ppg.unparsed_tokens[0] <<endl <<endl;
    usage(argc,argv);
  }
  if (param_config.filename.value.size()>0) {
    cerr << endl;
    cerr << "reading options from configuration file <" << param_config.filename.value << ">" <<endl<<endl;
    ppg.config_file_parse(param_config.filename.value);

    if (ppg.unparsed_tokens.size()>0) {
    cerr << "unknown option " << ppg.unparsed_tokens[0] <<endl <<endl;
    usage(argc,argv);
    }
    ppg.command_line_parse(argc,argv); 
  }
}

