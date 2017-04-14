FROM ubuntu:16.04

RUN	apt-get update

RUN	apt-get install -y software-properties-common && \
	apt-get install -y python-software-properties 

RUN	apt-get update && \
	apt-get install -y default-jdk 

RUN	apt-get install -y maven
	
RUN 	apt-get install -y python3-pip

RUN	apt-get install -y libssl-dev && \
	apt-get install -y libffi-dev && \
	apt-get install -y python3-dev && \
	apt-get install -y python3-venv 

RUN	apt-get install -y curl && \
	apt-get install -y git && \
	apt-get install -y iptables && \
	apt-get install -y less && \
	apt-get install -y vim && \
	apt-get install -y vim-common && \
	apt-get install -y tar && \
	apt-get install -y zip && \
	apt-get install -y unzip

RUN	apt-get install -y build-essential && \
 	apt-get install -y apt-utils && \
	apt-get install -y automake && \
	apt-get install -y cmake && \
	apt-get install -y libprotobuf-dev && \
	apt-get install -y gcc && \
	apt-get install -y gcc-4.9 && \
	apt-get install -y g++ && \
	apt-get install -y g++-4.9 && \
	apt-get install -y gcc-multilib && \
	apt-get install -y libgomp1 && \
	apt-get install -y pkg-config && \
	apt-get install -y sphinx-common && \
	apt-get install -y gfortran && \
	apt-get install -y maven 

RUN	apt-get install -y yasm  && \
	apt-get install -y libxext-dev  && \
	apt-get install -y libfreetype6-dev  && \
	apt-get install -y libsdl2-dev  && \
	apt-get install -y libtheora-dev  && \
	apt-get install -y libtool  && \
	apt-get install -y libva-dev  && \
	apt-get install -y libvdpau-dev  && \
	apt-get install -y libvorbis-dev  && \
	apt-get install -y libxcb1-dev  && \
	apt-get install -y libxcb-shm0-dev  && \
	apt-get install -y libxcb-xfixes0-dev  && \
	apt-get install -y zlib1g-dev  

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

RUN	apt-get install -y wget

RUN	apt-get install -y libblas-dev && \
	apt-get install -y liblapack-dev

RUN	apt-get install -y libreadline-dev && \
	apt-get install -y readline-common 
	
RUN	curl -R -O http://www.lua.org/ftp/lua-5.3.4.tar.gz && \
	tar zxf lua-5.3.4.tar.gz && \
	cd lua-5.3.4 && \
	make linux && \
	make install

RUN 	apt-get install -y python-dev && \
	apt-get install -y python-pip && \
	apt-get install -y python-scipy && \
	apt-get install -y python-numpy && \
	apt-get install -y python-matplotlib && \
	apt-get install -y python-nose 

RUN	apt-get install -y libpng12-dev && \
	apt-get install -y libjpeg-dev && \
        apt-get install -y libfreeimage-dev  && \
	apt-get install -y libfreetype6-dev 
