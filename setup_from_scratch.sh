# basic update on a newly created machine;
sudo apt update
sudo apt upgrade -y
# library needed later to run: gu rebuild-images polyglot and setting up graalpython;
sudo apt install build-essential -y
sudo apt install lib32z1-dev -y
sudo apt install unzip -y
sudo apt-get install -y python-ctypes

# clone repositories (GraalVM, MX, GrCUDA);
git clone https://github.com/oracle/graal.git
git clone https://github.com/graalvm/mx.git
git clone https://github.com/AlbertoParravicini/grcuda.git

# Checkout commit of GraalVM corresponding to the release;
cd graal
git checkout 192eaf62331679907449ee60dad9d6d6661a3dc8
cd ..

# Checkout commit of mx compatible with versions of other tools;
cd mx
git checkout dcfdd847c5b808ffd1a519713e0598242f28ecd1
cd ..

# download the GraalVM release build (21.1.0) and the corresponding JVM;
wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-21.1.0/graalvm-ce-java11-linux-amd64-21.1.0.tar.gz
wget https://github.com/graalvm/labs-openjdk-11/releases/download/jvmci-21.1-b05/labsjdk-ce-11.0.11+8-jvmci-21.1-b05-linux-amd64.tar.gz
# extract them;
tar xfz labsjdk-ce-11.0.11+8-jvmci-21.1-b05-linux-amd64.tar.gz
tar xfz graalvm-ce-java11-linux-amd64-21.1.0.tar.gz
# remove temporary files;
rm labsjdk-ce-11.0.11+8-jvmci-21.1-b05-linux-amd64.tar.gz
rm graalvm-ce-java11-linux-amd64-21.1.0.tar.gz

# install CUDA and Nvidia drivers;

# option 1 (more automatic, but possibly outdated);
# sudo apt install nvidia-cuda-toolkit -y
# sudo apt install ubuntu-drivers-common -y
# sudo ubuntu-drivers autoinstall
# option 2 (from Nvidia's website);
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# symlink for python (use it with care! some system tools require Python 2.7);
# sudo ln -s /usr/bin/python3 /usr/bin/python

# update ~/.bashrc with new variables;
echo '' >> ~/.bashrc
echo '########## GrCUDA Configuration ##########' >> ~/.bashrc
echo '' >> ~/.bashrc
echo '# CUDA;' >> ~/.bashrc
echo 'export CUDA_DIR=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_DIR/bin:$PATH' >> ~/.bashrc
echo '# GraalVM and GrCUDA;' >> ~/.bashrc
echo 'export PATH=~/mx:$PATH' >> ~/.bashrc
echo 'export JAVA_HOME=~/labsjdk-ce-11.0.11-jvmci-21.1-b05' >> ~/.bashrc
echo 'export GRAAL_HOME=~/graalvm-ce-java11-21.1.0' >> ~/.bashrc
echo 'export PATH=$GRAAL_HOME/bin:$PATH' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export GRCUDA_HOME=~/grcuda' >> ~/.bashrc
echo '' >> ~/.bashrc
echo '##########################################' >> ~/.bashrc
# reload  ~/.bashrc;
source  ~/.bashrc

# setup GraalVM;
gu install native-image
gu install llvm-toolchain
gu install python 
gu rebuild-images polyglot

# create environment for Graalpython and set it up;
graalpython -m venv ~/graalpython_venv
source ~/graalpython_venv/bin/activate
graalpython -m ginstall install setuptools
graalpython -m ginstall install Cython
graalpython -m ginstall install numpy

# reboot the machine to load the Nvidia drivers;
sudo reboot

