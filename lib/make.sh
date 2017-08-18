TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')


CXXFLAGS=''

cd reorg_layer

if [ -d "$CUDA_PATH" ]; then
    /usr/local/cuda/bin/nvcc -std=c++11 -c -o reorg_op.cu.o reorg_op_gpu.cu.cc \
        -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
        -arch=sm_61 --expt-relaxed-constexpr
            
    g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o reorg.so reorg_op.cc \
        reorg_op.cu.o -I $TF_INC -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
        -lcudart -L $CUDA_PATH/lib64
else
    g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o reorg.so reorg_op.cc \
        -I $TF_INC -fPIC $CXXFLAGS
fi

cd ..

cd region_layer

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o region.so region_op.cc \
    -I $TF_INC -fPIC $CXXFLAGS
