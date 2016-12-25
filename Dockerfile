FROM yandex/rep:0.6.4
MAINTAINER Soboleva Daria <daria_soboleva_vmk@gmail.com>



RUN apt-get update && apt-get install -y \
    git \
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
    build-essential
    
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

RUN pip install tqdm
RUN pip install protobuf

RUN bash --login -c "source activate jupyterhub_py3 && pip install\
  
  python-telegram-bot\
  pandas\
  numpy\
  seaborn\
  sklearn\
  matplotlib"
