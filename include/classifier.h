/************************************************************************
 *  classifier.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _RGF_CLASSIFIER_H

#define _RGF_CLASSIFIER_H

#include "data.h"

namespace rgf {

  
  namespace TrainLoss {
    enum {
      LS=0,   
      MODLS=1, 
      LOGISTIC=2, 
      INVALID =3  
    };

    
    int str2loss(string loss_str);

    
    string loss2str(int loss);

    
    double binary_loss(int loss, double scr, double y);
  }

  
  template<typename d_t, typename i_t, typename v_t>
  class BinaryClassifier {
  public:
    
    double threshold;

    
  BinaryClassifier() :
    threshold(0.0) {
    }

    
    virtual double apply(DataPoint<d_t,i_t,v_t> & dp)=0;

    
    bool classify(double scr) {
      return scr > threshold;
    }

    virtual ~BinaryClassifier() {}
  };

  
  class BinaryTestStat {
    
    class TestResult {
    public:
      
      double scr;
      
      double y;
      
      const bool operator<(const TestResult & b) const {
	return scr < b.scr;
      }

      
    TestResult(double _scr, double _y) :
      scr(_scr), y(_y) {
      }
    };

    
    vector<TestResult> _results;

    
    Target _y_type;

    
    int _loss;
  public:
    
    size_t tp;
    
    size_t tn;
    
    size_t fp;
    
    size_t fn;

    
    size_t num;

    
    double total_loss;

    
    bool keep_results;

    
    BinaryTestStat(Target y_type, int loss) :
      _y_type(y_type), _loss(loss), tp(0), tn(0), fp(0), fn(0),
      num(0), total_loss(0), keep_results(true) {
    }

    
    void update(double y, double scr, bool pred_label);

    
    template<typename d_t, typename i_t, typename v_t>
    void update(BinaryClassifier<d_t,i_t,v_t> & appl, DataSet<d_t,i_t,v_t> & ds)
    {
      for (size_t i=0; i<ds.size(); i++) {
	DataPoint<d_t,i_t,v_t> dp=ds[i];
	double scr=appl.apply(dp);
	update(ds.y[i], scr, appl.classify(scr));
      }
    }


    
    double accuracy() {
      return (tp + tn) / (tp + tn + fp + fn + 1e-10);
    }
    
    double precision() {
      return tp / (tp + fp + 1e-10);
    }
    
    double recall() {
      return tp / (tp + fn + 1e-10);
    }
    
    double fb1() {
      return 2.0 / (1.0 / precision() + 1.0 / recall());
    }

    
    void roc(size_t _tp, size_t _tn, double & tpr, double & fpr);

    
    double auc();

    
    double mse();

    
    void print(ostream & os);

    
    void clear();
  };

  
}

#endif

