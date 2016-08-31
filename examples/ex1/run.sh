#!/bin/sh -f
#

exe_train=../../bin/forest_train
exe_predict=../../bin/forest_predict

trn=inputs/madelon.train

tst=inputs/madelon.test

model_rgf=outputs/model-rgf
    
prediction=outputs/prediction

orig_format="y.sparse"
save_freq=200

echo ------ training ------
time ${exe_train} trn.x-file=${trn} trn.x-file_format=${orig_format} trn.target=BINARY tst.x-file=${tst} tst.x-file_format=${orig_format} tst.target=BINARY model.save=${model_rgf} dtree.new_tree_gain_ratio=1.0  dtree.lamL2=5000  forest.ntrees=1000  dtree.loss=LOGISTIC forest.save_frequency=${save_freq}

echo " "
echo " "
echo ------ testing intermediate model at ${save_freq} on ${tst} ------
time ${exe_predict} tst.x-file=${tst} tst.x-file_format=${orig_format} tst.target=BINARY model.load=${model_rgf}-${save_freq} 

echo " "
echo " "
echo ------ testing ------
for datafile in ${trn}  ${tst}
do
   suffix=`echo ${datafile}|sed 's/.*\.//g'`
   echo === $datafile ===	
   time ${exe_predict} tst.x-file=${datafile} tst.x-file_format=${orig_format} tst.target=BINARY model.load=${model_rgf} tst.output-prediction=${prediction}-${suffix} 
done
    

