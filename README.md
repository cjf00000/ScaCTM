Scalable inference for Correlated Topic Models (ScaCTM)
====

Install
----

Assume you already have BLAS and LAPACK installed, to build ScaCTM, simply do

        cd third_party
        make                    # Install third-party dependencies, takes 6 minutes on our 12-core server
        cd ..
        cd src
        make -j                 # Build ScaCTM in parallel

See [BLAS](#BLAS) if you don't have BLAS and LAPACK.

Usage
----
### Input data
        wget https://raw.github.com/sudar/Yahoo_LDA/master/test/ydir_1k.txt

### Single machine usage

### Distributed usage

### Output data

BLAS
----

If you don't have BLAS and LAPACK, you can simply install them by package manager, for example

        sudo apt-get install libblas-dev
        sudo apt-get install liblapack-dev

will install [Netlib reference implementation](http://www.netlib.org/blas/) on Ubuntu systems.

If you are working with a lot of short documents and many topics (>=500), you may wish to use a faster BLAS. There are many possible options, for example, to install [ATLAS](http://math-atlas.sourceforge.net/)

        sudo apt-get install libatlas3-dev

and rebuild the code

        cd src
        make clean && make -j

make sure ATLAS is used by ScaCTM

        # See which BLAS is ScaCTM using
        $ ldd /bin/learntopics
        <skipped>
        libblas.so.3gf => /usr/lib/libblas.so.3gf (0x00007f7c770e0000)
        <skipped>

        # Find the path of /usr/lib/libblas.so.3gf
        $ ls -l /usr/lib/libblas.so.3gf
        /usr/lib/libblas.so.3gf -> /etc/alternatives/libblas.so.3gf 

        $ ls -l /etc/alternatives/libblas.so.3gf
        /etc/alternatives/libblas.so.3gf -> /usr/lib/atlas-base/atlas/libblas.so.3gf

if you are still using the old BLAS, you can manually link with ATLAS by modifing LDFLAGS in src/Makefile.

### NOTE

Be cautious with OpenBLAS, we find ScaCTM works extremely slow with OpenBLAS.

Dependencies
----

ScaCTM is built upon the Yahoo-LDA, which is a distributed framework for topic modeling.

License
----

Apache License 2.0

Reference
----

Jianfei Chen, Jun Zhu, Zi Wang, Xun Zheng, and Bo Zhang. Scalable Inference for Logistic-Normal Topic Models, Advances in Neural Information Processing Systems (NIPS), Lake Tahoe, USA, 2013. (NIPS 2013)

Some demonstration can be found at http://ml-thu.net/~scalable-ctm/.
