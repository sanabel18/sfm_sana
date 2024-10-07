FROM registry.corp.ailabs.tw/smartcity/video-processing/base/v2-5-0/cpu:latest
MAINTAINER hkazami@ailabs.tw

# time, locale, and basic utilities

RUN echo 'Asia/Taipei' > /etc/timezone \
 && echo locales locales/default_environment_locale select en_US.UTF-8 | debconf-set-selections \
 && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      tzdata \
      locales \
      locales-all \
      \
      vim \
      htop \
      tmux \
 && echo "PS1='[\[\033[01;36m\]\\\\t\[\033[00m\]] \u@\[\033[01;32m\]\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc \
 && echo export LANG=en_US.UTF-8 >> /root/.bashrc \
 && echo export LC_ALL=en_US.UTF-8 >> /root/.bashrc \
 && echo "alias tm='tmux -CC'" >> /root/.bashrc \
 && echo "alias tma='tmux -CC attach'" >> /root/.bashrc


# ---------------------------------------------------------------------------
# For recon3d package

COPY requirements.txt /root/
RUN pip install -r /root/requirements.txt
RUN pip --no-cache-dir install \
      git+https://gitlab+deploy-token-375:A3XmLPZoL7DyYSYyRM6g@gitlab.corp.ailabs.tw/smart-city/cpl-platform/video-processing/videoio.git@v2.0.1 \
      git+https://gitlab+deploy-token-5:ysm-5JisLfZcEFDQ6-L9@gitlab.corp.ailabs.tw/smart-city/cpl-platform/video-processing/360-utils.git@v1.4.1 \
      git+https://gitlab+deploy-token-1076:MAuGvJXaBxS4cbygnPD-@gitlab.corp.ailabs.tw/smartcity/reconstruction/votool.git@v1.4.1
RUN pip install utm

RUN conda install -y \
      toml \
      pytz \
      scikit-learn

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      xvfb

RUN conda install -y -c open3d-admin open3d

############### Quick fix of Open3D missing addict error ################
RUN pip install addict pandas plyfile tqdm scikit-learn matplotlib
########## See https://github.com/intel-isl/Open3D/issues/2504 ##########

# ---------------------------------------------------------------------------
# Meshlab

# Setup Qt environment in order to build MeshLab
RUN apt-get install -y \
      qt5-default \
      libqt5xmlpatterns5-dev \
      qtdeclarative5-dev

# Build MeshLab 2020.07
RUN cd /opt \
 && git clone --recursive --depth 1 --branch Meshlab-2020.07 https://github.com/cnr-isti-vclab/meshlab \
 && cd meshlab \
 && cmake src/ \
 && make -j6


# ---------------------------------------------------------------------------
# OpenMVG

ENV PATH $PATH:/opt/openmvg_build/install/bin
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      graphviz \
      coinor-libclp-dev \
      libceres-dev \
      libflann-dev \
      liblemon-dev \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev

ADD openmvg /opt/openmvg

RUN mkdir /opt/openmvg_build \
 && cd /opt/openmvg_build \
 && cmake -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX="/opt/openmvg_build/install" \
    -DOpenMVG_BUILD_TESTS=ON \
    -DOpenMVG_BUILD_EXAMPLES=OFF \
    -DFLANN_INCLUDE_DIR_HINTS=/usr/include/flann \
    -DLEMON_INCLUDE_DIR_HINTS=/usr/include/lemon \
    -DCOINUTILS_INCLUDE_DIR_HINTS=/usr/include \
    -DCLP_INCLUDE_DIR_HINTS=/usr/include \
    -DOSI_INCLUDE_DIR_HINTS=/usr/include \
    ../openmvg/src \
 && make -j6 \
 && cd /opt/openmvg_build \
 && make test \
 && make install


# ---------------------------------------------------------------------------
# OpenMVS

ENV PATH /usr/local/bin/OpenMVS:$PATH
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      mercurial \
      libglu1-mesa-dev \
      libxmu-dev \
      libxi-dev \
      \
      libboost-iostreams-dev \
      libboost-program-options-dev \
      libboost-system-dev \
      libboost-serialization-dev \
      \
      libcgal-dev \
      libcgal-qt5-dev \
      \
      libatlas-base-dev \
      libsuitesparse-dev \
      freeglut3-dev \
      libglew-dev \
      libglfw3-dev

RUN cd /opt \
 && git clone https://github.com/cdcseacave/VCG.git vcglib
RUN cd /opt/vcglib \
 && git reset --hard 88f12f212a1645d1fa6416592a434c29e63b57f0
ADD openmvs /opt/openmvs

RUN mkdir /opt/openmvs_build \
 && cd /opt/openmvs_build \
 && cmake . ../openmvs -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT=/opt/vcglib \
 && make -j6 \
 && make install


# ---------------------------------------------------------------------------
# Installations for Tower

RUN pip install --ignore-installed PyYAML
RUN pip install git+https://gitlab+deploy-token-579:F9z4f9ud9Gms1muzFXjX@gitlab.corp.ailabs.tw/smart-city/cpl-platform/platform/tower-cli.git
RUN pip install git+https://gitlab+deploy-token-899:FUKA6BasiXzzHqssn33v@gitlab.corp.ailabs.tw/smartcity/reconstruction/RTK2geojson.git@v1.1.0
RUN pip install git+https://gitlab+deploy-token-796:Lrx45tFmFHgZrsFBbb-y@gitlab.corp.ailabs.tw/smartcity/reconstruction/pytower.git@v1.1.3
# ---------------------------------------------------------------------------
# Finalize

COPY src /root/src/
COPY openmvg/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt /root/src/
RUN chmod 755 /root/src/*.py
# Add PYTHONPATH
ENV PYTHONPATH="/root/src/recon3d:${PYTHONPATH}"
RUN mkdir /root/proj
WORKDIR /root/src
