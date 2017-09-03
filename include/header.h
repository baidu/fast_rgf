/************************************************************************
 *  header.h (2016) by Tong Zhang
 *  
 *  For Copyright, see LICENSE.
 *  
 ************************************************************************/


#ifndef _RGF_HEADER_H

#define _RGF_HEADER_H

#include <cassert>

#include <string>
#include <queue>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <cmath>

#include <cstring>

#ifdef USE_OMP
#include "omp.h"
#endif

namespace rgf {
  

#define VER "version 0.4 (Aug 2017) by Tong Zhang"

  
  const int max_thrds=128;

  
  const int vect_width=8;

  
  using int_t=long;


  
  using src_index_t=int;

  
  using disc_dense_value_t=unsigned short;

  
  using disc_sparse_index_t=int;

  
  using disc_sparse_value_t=unsigned char;
  
  
  
  using train_size_t=unsigned int;
}

using namespace std;
using namespace rgf;

#endif













