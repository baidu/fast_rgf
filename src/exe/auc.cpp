/************************************************************************
 *  auc.cpp (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#include "classifier.h"

int main(int argc, char *argv[])
{
  if (argc>1 && argv[1][0]=='-' && argv[1][1]=='h') {
    cout << "usage: " << argv[0] << " < result-file " <<endl;
    cout << " result-file format: true-label predicted-score" <<endl;
    exit(-1);
  }

  Target y_type("BINARY");
  int loss=TrainLoss::str2loss(string("LS"));
  BinaryTestStat test_result(y_type,loss);

  int nl=0;
  while(true) {
    bool label;
    double y;
    double score;
    cin >> y >> score;
    label=y_type.binary_label(y);
    if (cin.eof()) break;
    test_result.update(label,0,score);
    nl++;
  }
  cout << "auc=" << test_result.auc() <<endl;
}
