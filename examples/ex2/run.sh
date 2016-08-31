#!/bin/sh -f
#

exe_train=../../bin/forest_train
exe_predict=../../bin/forest_predict

trn=inputs/housing.train
tst=inputs/housing.test
feat_name=inputs/feature.names
    
config=inputs/config

model_rgf=outputs/model-rgf
    
prediction=outputs/prediction

orig_format="y.x"

echo ------ training ------
time ${exe_train} -config=${config} trn.x-file=${trn} trn.x-file_format=${orig_format}  trn.target=REAL tst.x-file=${tst} tst.x-file_format=${orig_format} tst.target=REAL model.save=${model_rgf} 

echo ------ print forest ------    
${exe_predict} model.load=${model_rgf} tst.print-forest=${model_rgf}.print tst.feature-names=${feat_name}

echo " "
echo " "
echo ------ testing ------
for datafile in ${trn}  ${tst}
do
   suffix=`echo ${datafile}|sed 's/.*\.//g'`
   echo === $datafile ===	
   time ${exe_predict} tst.x-file=${datafile} tst.x-file_format=${orig_format} tst.target=REAL model.load=${model_rgf} tst.output-prediction=${prediction}-${suffix} 
done
    
    
