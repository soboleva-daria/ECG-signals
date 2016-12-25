FROM yandex/rep:0.6.4
MAINTAINER Soboleva Daria <daria_soboleva_vmk@gmail.com>


RUN apt-get update && apt-get install -y \
    git\
    python-dev\
    libxml2-dev\
    libxslt1-dev\
    zlib1g-dev\
    cmake\
    make\
    libreadline-dev\
    luarocks\
    libprotobuf-dev\
    protobuf-compiler\
    libboost-all-dev\
    python-pip\
    build-essential\
    
RUN pip install setuptools
RUN pip install protobuf
RUN pip install tqdm

RUN wget https://bootstrap.pypa.io/get-pip.py 
RUN python get-pip.py
    
RUN git clone --branch=stable https://github.com/bigartm/bigartm.git\
   && (cd bigartm && mkdir build && cd build && cmake .. && make install )
   
RUN bash export ARTM_SHARED_LIBRARY=/usr/local/lib/libartm.so

RUN bash --login -c "source activate jupyterhub_py3 && pip install\
  
  python-telegram-bot\
  pandas\
  numpy\
  seaborn\
  sklearn\
  matplotlib"
