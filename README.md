----------
#                                       FastRGF
###     Multi-core implementation of Regularized Greedy Forest [RGF] 

### Version 0.3 (Dec 2016) by Tong Zhang
                          
---------
#### 1. Introduction

This software package provides a multi-core implementation of a simplified Regularized Greedy Forest (RGF) described in **[RGF]**. Please cite the paper if you find the software useful. 

RGF is a machine learning method for building decision forests that have been used to win some kaggle competitions. In our experience it works better than *gradient boosting* on many relatively large datasets.

The implementation employs the following conepts described in the **[RGF]** paper:

-  tree node regularization
-  fully-corrective update
-  greedy node expansion with trade-off between leaf node splitting for current tree and root splitting for new tree

However, various simplifications are made to accelerate the training speed. Therefore, unlike the original RGF program (see <http://stat.rutgers.edu/home/tzhang/software/rgf/>), this software does not reproduce the results in the paper. 

The implementation of greedy tree node optimization employs second order Newton approximation for general loss functions. For logistic regression loss, which works especially well for many binary classification problems, this approach was considered in **[PL]**; for general loss functions, 2nd order approximation was considered in **[ZCS]**.

#### 2. Installation
Please see the file [CHANGES](CHANGES) for version information.
The software is written in c++11, and it has been tested under linux and macos, and it may require g++ version 4.8 or above and cmake version 2.8 or above. 
 

 To install the binaries, unpackage the software into a directory.
 
 * The source files are in the subdirectories include/ and src/.
 * The executables are under the subdirectory bin/.
 * The examples are under the subdirectory examples/.

 To create the executables, do the following:
 
     cd build/
     cmake ..
     make 
     make install

 The following executabels will be installed under the subdirectory bin/. 
 
* forest_train: train rgf and save model
* forest_predict: apply trained model on test data 

You may use the option -h to show command-line options (options can also be provided in a configuration file).
 
#### 3. Examples
 Go to the subdirectory examples/, and following the instructions in [README.md](examples/README.md). The file also contains some tips for parameter tuning.
 
#### 4. Contact
Tong Zhang

#### 5. Copyright
The software is distributed under the MIT license. Please read the file [LICENSE](LICENSE).

#### 6. References

**[RGF]** Rie Johnson and Tong Zhang. [Learning Nonlinear Functions Using Regularized Greedy Forest](http://arxiv.org/abs/1109.0887), *IEEE Trans. on Pattern Analysis and Machine Intelligence, 36:942-954*, 2014.

**[PL]** Ping Li. Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost, *UAI* 2010.

**[ZCS]** Zhaohui Zheng, Hongyuan Zha, Tong Zhang, Olivier Chapelle, Keke Chen, Gordon Sun. A general boosting method and its application to learning ranking functions for web search, *NIPS* 2007.

