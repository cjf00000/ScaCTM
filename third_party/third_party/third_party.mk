# boost is too heavy for git to host...
THIRD_PARTY_HOST = http://github.com/xunzheng/third_party/raw/master
BOOST_HOST = http://downloads.sourceforge.net/project/boost/boost/1.54.0

third_party: gflags \
             glog \
             protobuf \
             gperftools \
             boost \
             tbb \
             ice \
	     armadillo \
	     mpich

.PHONY: third_party

# ===================== gflags ===================

GFLAGS_SRC = $(THIRD_PARTY_SRC)/gflags-2.0.tar.gz
GFLAGS_LIB = $(THIRD_PARTY_LIB)/libgflags.so

gflags: path $(GFLAGS_LIB)

$(GFLAGS_LIB): $(GFLAGS_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY); \
	make -j install

$(GFLAGS_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ===================== glog =====================

GLOG_SRC = $(THIRD_PARTY_SRC)/glog-0.3.3.tar.gz
GLOG_LIB = $(THIRD_PARTY_LIB)/libglog.so

glog: path gflags $(GLOG_LIB)

$(GLOG_LIB): $(GLOG_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY) --with-gflags=$(THIRD_PARTY); \
	make -j install

$(GLOG_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ===================== gtest ====================

GTEST_SRC = $(THIRD_PARTY_SRC)/gtest-1.7.0.zip
GTEST_LIB = $(THIRD_PARTY_LIB)/libgtest_main.a

gtest: path $(GTEST_LIB)

$(GTEST_LIB): $(GTEST_SRC)
	unzip $< -d $(THIRD_PARTY_SRC)
	cd $(basename $<)/make; \
	make -j; \
	./sample1_unittest; \
	cp -r ../include/* $(THIRD_PARTY_INCLUDE)/; \
	cp gtest_main.a $@

$(GTEST_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ==================== zeromq ====================
# NOTE: need uuid-dev

ZMQ_SRC = $(THIRD_PARTY_SRC)/zeromq-3.2.3.tar.gz
ZMQ_LIB = $(THIRD_PARTY_LIB)/libzmq.so

zeromq: path $(ZMQ_LIB)

$(ZMQ_LIB): $(ZMQ_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY); \
	make -j install

$(ZMQ_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@
	wget $(THIRD_PARTY_HOST)/zmq.hpp -P $(THIRD_PARTY_INCLUDE)

# ==================== boost ====================

BOOST_SRC = $(THIRD_PARTY_SRC)/boost_1_54_0.tar.bz2
BOOST_INCLUDE = $(THIRD_PARTY_INCLUDE)/boost

boost: path $(BOOST_INCLUDE)

$(BOOST_INCLUDE): $(BOOST_SRC)
	tar jxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./bootstrap.sh --with-libraries=system,thread --prefix=$(THIRD_PARTY); \
	./b2 install

$(BOOST_SRC):
	wget $(BOOST_HOST)/$(@F) -O $@

# ================== gperftools =================

GPERFTOOLS_SRC = $(THIRD_PARTY_SRC)/gperftools-2.1.tar.gz
GPERFTOOLS_LIB = $(THIRD_PARTY_LIB)/libtcmalloc.so

gperftools: path $(GPERFTOOLS_LIB)

$(GPERFTOOLS_LIB): $(GPERFTOOLS_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY) --enable-frame-pointers; \
	make -j install

$(GPERFTOOLS_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ===================== tbb =====================

TBB_SRC = $(THIRD_PARTY_SRC)/tbb42_20130725oss.tgz
TBB_LIB = $(THIRD_PARTY_LIB)/libtbb.so

tbb: path $(TBB_LIB)

$(TBB_LIB): $(TBB_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $<); \
	make -j; \
	cp build/*_release/lib* $(THIRD_PARTY_LIB)/; \
	cp -r include/tbb $(THIRD_PARTY_INCLUDE)/

$(TBB_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ================== sparsehash ==================

SPARSEHASH_SRC = $(THIRD_PARTY_SRC)/sparsehash-2.0.2.tar.gz
SPARSEHASH_INCLUDE = $(THIRD_PARTY_INCLUDE)/sparsehash

sparsehash: path $(SPARSEHASH_INCLUDE)

$(SPARSEHASH_INCLUDE): $(SPARSEHASH_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY); \
	make -j install

$(SPARSEHASH_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# =================== oprofile ===================
# NOTE: need libpopt-dev binutils-dev

OPROFILE_SRC = $(THIRD_PARTY_SRC)/oprofile-0.9.9.tar.gz
OPROFILE_LIB = $(THIRD_PARTY_LIB)/oprofile

oprofile: path $(OPROFILE_LIB)

$(OPROFILE_LIB): $(OPROFILE_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY); \
	make -j install

$(OPROFILE_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ================== protobuf ==================

PROTOBUF_SRC = $(THIRD_PARTY_SRC)/protobuf-2.5.0.tar.bz2
PROTOBUF_LIB = $(THIRD_PARTY_LIB)/libprotobuf.so

protobuf: path $(PROTOBUF_LIB)

$(PROTOBUF_LIB): $(PROTOBUF_SRC)
	tar jxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY); \
	make -j install

$(PROTOBUF_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ==================== mcpp ====================
# NOTE: this is Ice patched version.
# See http://www.zeroc.com/download/Ice/3.5/ThirdParty-Sources-3.5.1.tar.gz

MCPP_SRC = $(THIRD_PARTY_SRC)/mcpp-2.7.2.tar.gz
MCPP_LIB = $(THIRD_PARTY_LIB)/libmcpp.a

mcpp: path $(MCPP_LIB)

$(MCPP_LIB): $(MCPP_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure CFLAGS=-fPIC \
                    --enable-mcpplib \
                    --disable-shared \
                    --prefix=$(THIRD_PARTY); \
	make -j install

$(MCPP_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# =================== bzip2 ====================

BZIP2_SRC = $(THIRD_PARTY_SRC)/bzip2-1.0.6.tar.gz
BZIP2_LIB = $(THIRD_PARTY_LIB)/libbz2.a

bzip2: path $(BZIP2_LIB)

$(BZIP2_LIB): $(BZIP2_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	make -j install PREFIX=$(THIRD_PARTY) \
                     CFLAGS='-O4 -D_FILE_OFFSET_BITS=64 -fPIC'

$(BZIP2_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ==================== Ice =====================

ICE_SRC = $(THIRD_PARTY_SRC)/Ice-3.5.1.tar.gz
ICE_LIB = $(THIRD_PARTY_LIB)/libIce.so

ice: path mcpp bzip2 $(ICE_LIB)

$(ICE_LIB): $(ICE_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	-mkdir $(THIRD_PARTY)/lib64
	cp $(THIRD_PARTY_LIB)/libmcpp.a $(THIRD_PARTY)/lib64
	cd $(basename $(basename $<))/cpp; \
	sed -i '14c SUBDIRS=config src include'             Makefile; \
	sed -i '14,28c SUBDIRS=Ice IceUtil Slice'           include/Makefile; \
	sed -i '22,48c SUBDIRS=IceUtil Slice slice2cpp Ice' src/Makefile; \
	sed -i "14c prefix=$(THIRD_PARTY)"                  config/Make.rules; \
	sed -i "20c embedded_runpath_prefix=$(THIRD_PARTY)" config/Make.rules; \
	sed -i '33c OPTIMIZE=yes'                           config/Make.rules; \
	sed -i "76c BZIP2_HOME=$(THIRD_PARTY)"              config/Make.rules; \
	sed -i "102c MCPP_HOME=$(THIRD_PARTY)"              config/Make.rules; \
	make -j install
	mv $(THIRD_PARTY)/lib64/* $(THIRD_PARTY_LIB)
	rm -rf $(THIRD_PARTY)/lib64

$(ICE_SRC):
	wget $(THIRD_PARTY_HOST)/$(@F) -O $@

# ===================== Armadillo =================
ARMA_SRC = $(THIRD_PARTY_SRC)/armadillo-3.930.2.tar.gz
ARMA_LIB = $(THIRD_PARTY_INCLUDE)/armadillo

armadillo: $(ARMA_LIB)

$(ARMA_LIB): $(ARMA_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	cp -r include/* $(THIRD_PARTY_INCLUDE)
	sed -i '25c \ ' $(THIRD_PARTY_INCLUDE)/armadillo_bits/config.hpp

$(ARMA_SRC):
	wget http://ml-thu.net/~jianfei/static/dependencies/armadillo-3.930.2.tar.gz -O $@

# ===================== OpenBLAS ==================
OPENBLAS_SRC = $(THIRD_PARTY_SRC)/OpenBLAS.tar.gz
OPENBLAS_LIB = $(THIRD_PARTY_LIB)/libopenblas.so

openblas: $(OPENBLAS_LIB)

$(OPENBLAS_LIB): $(OPENBLAS_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	make -j; \
	make install PREFIX=$(THIRD_PARTY)

$(OPENBLAS_SRC):
	wget http://ml-thu.net/~jianfei/static/dependencies/OpenBLAS.tar.gz -O $@

# ===================== MPI =======================
MPI_SRC = $(THIRD_PARTY_SRC)/mpich-3.0.4.tar.gz
MPI_LIB = $(THIRD_PARTY_LIB)/libmpichcxx.a

mpich: $(MPI_LIB)

$(MPI_LIB): $(MPI_SRC)
	tar zxf $< -C $(THIRD_PARTY_SRC)
	cd $(basename $(basename $<)); \
	./configure --prefix=$(THIRD_PARTY); \
	make -j 12; \
	make install

$(MPI_SRC):
	wget http://ml-thu.net/~jianfei/static/dependencies/mpich-3.0.4.tar.gz -O $@
