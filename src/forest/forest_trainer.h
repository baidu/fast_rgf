/************************************************************************
 *  forest_trainer.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "dtree.h"

namespace rgf {
  
  class DecisionForestTrainer
  {
    
    void * trainer_ptr=nullptr;

    
    bool _is_fully_corrective;
  public:

    DecisionForestTrainer(string opt="") {
      if (opt.compare("rgf")  && opt.compare("epsilon-greedy")) {
	cerr << " invalid forest training method " << opt << endl;
	cerr << " valid values are rgf or epsilon-greedy" <<endl;
	exit(-1);
      }
      _is_fully_corrective= (opt.compare("rgf")==0);
    }

    bool is_fully_corrective() {
      return _is_fully_corrective;
    }
    
    
    template<typename d_t, typename i_t, typename v_t>
    void init(DataSet<d_t,i_t,v_t> & ds, int ngrps, int verbose);

    
    template<typename d_t, typename i_t, typename v_t>
    void build_single_tree(DataSet<d_t,i_t,v_t> & ds, double *scr_arr, 
			   class DecisionTree<d_t,i_t,v_t>::TrainParam & param_dt,
			   double step_size,
			   class DecisionTree<d_t,i_t,v_t> & dtree);

    template<typename d_t, typename i_t, typename v_t>
    void fully_corrective_update(DataSet<d_t,i_t,v_t> & ds, double *scr_arr, 
				 class DecisionTree<d_t,i_t,v_t>::TrainParam &param_dt,
				 DecisionTree<d_t,i_t,v_t> * dtree_vec,
				 int ntrees);

    
    template<typename d_t, typename i_t, typename v_t>
    void finish(DataSet<d_t,i_t,v_t> &ds, int verbose);
  };
}
