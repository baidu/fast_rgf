# create test programs forest_train forest_predict
#

add_executable(auc auc.cpp) 
target_link_libraries(auc base)

#install(TARGETS auc DESTINATION ${CMAKE_SOURCE_DIR}/bin)

add_executable(discretized_gendata discretized_gendata.cpp) 
target_link_libraries(discretized_gendata base)

#install(TARGETS discretized_gendata DESTINATION ${CMAKE_SOURCE_DIR}/bin)


add_executable(discretized_trainer discretized_trainer.cpp) 
target_link_libraries(discretized_trainer base forest)

#install(TARGETS discretized_trainer DESTINATION ${CMAKE_SOURCE_DIR}/bin)


############### 
add_executable(forest_train forest_train.cpp) 
target_link_libraries(forest_train forest base)

install(TARGETS forest_train DESTINATION ${CMAKE_SOURCE_DIR}/bin)

add_executable(forest_predict forest_predict.cpp) 
target_link_libraries(forest_predict forest base)

install(TARGETS forest_predict DESTINATION ${CMAKE_SOURCE_DIR}/bin)
