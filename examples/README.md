### Examples
---
* ex1 This is a binary classification problem, in libsvm's sparse feature format.
Use the *shell script* [run.sh](ex1/run.sh) to perform training/test.
The dataset is downloaded from <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#madelon>.
    
    
* ex2: This is a regression problem, in dense feature format. Use the *shell script* [run.sh](ex2/run.sh) to perform training/test.
The dataset is from <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing>.
    

Note that for these small examples, the running time with multi-threads may be slower than with single-thread due to the overhead it introduces. However, for large datasets, one can observe an almost linear speed up.

The program can directly handle high dimensional sparse features in the libsvm format as in ex1. This is the recommended format to use when the dataset is relatively large (although some other formats are supported).

---
### Tips for Parameter Tuning

There are multiple training parameters that can affect performance. The following are the more important ones:

* **dtree.loss**: default is LS, but for binary classificaiton, LOGISTIC often works better.
* **forest.ntrees**: typical range is [100,10000], and a typical value is 1000.
* **dtree.lamL2**: use a relatively large vale such as 1000 or 10000. The larger dtree.lamL2 is, the larger forest.ntrees you need to use: the resulting accuracy is often better with a longer training time.
* **dtree.lamL1**: try values in [0,1000], and a large value induces sparsity.
* **dtree.max_level** and **dtree.max_nodes** and **dtree.new_tree_gain_ratio**: these parameters control the tree depth and size (and when to start a new tree). One can try different values (such as dtree.max_level=4, or dtree.max_nodes=10, or dtree.new_tree_gain_ratio=0.5) to fine tuning performance.

You may also modify the discreitzation options below:

* **discretize.dense.max_buckets**: try in the range of [10,65000]
* **discretize.sparse.max_buckets**: try in the range of [10, 250]. If you want to try a larger value up to 65000, then you need to edit [../include/header.h](../include/header.h) and replace
 "*using disc_sparse_value_t=unsigned char;*"
    by "*using disc_sparse_value_t=unsigned short;*". However, this increase the memory useage.     
* **discretize.sparse.max_features**: you may try a different value in [1000,10000000].

