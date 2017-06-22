#!/usr/bin/bash

echo "Installing necessary packages..."

sudo apt update && apt upgrade
sudo apt install python-pip python3-pip unzip build-essential cmake pkg-config \
 libjpeg8-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev \
 libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libatlas-base-dev \
  gfortran python3.6-dev libtesseract3 libtesseract-dev liblept5 libleptonica-dev

echo "Downloading OpenCV..."
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip
unzip opencv.zip

echo "Downloading OpenCV Contribs..."
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip
unzip opencv_contrib.zip

echo "Creating virtualenv cv..."
echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc

pip install -U pip
pip3 install -U pip
pip install -U virtualenv virtualenvwrapper

mkvirtualenv cv -p python3
workon cv

pip3 install -U numpy

echo "Configuring OpenCV. Make sure it says TESSERACT: YES and that the Python3 paths are correct."
cd ~/opencv-3.2.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D INSTALL_PYTHON_EXAMPLES=OFF \
  -D INSTALL_C_EXAMPLES=OFF \
  -D ENABLE_PRECOMPILED_HEADERS=OFF \
  -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/OpenCV/opencv_contrib-3.2.0/modules \
  -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
  -D BUILD_EXAMPLES=OFF ..

echo "Continue?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

echo "Compiling and installing..."
make -j4
sudo make install
sudo ldconfig

echo "Configuring OpenCV for Python3..."
cd /usr/local/lib/python3.6/site-packages/
ls
echo "Please find the opencv .so file and enter its name here: "
read opencv_so
sudo mv $opencv_so cv2.so
cd ~/.virtualenvs/cv/lib/python3.6/site-packages/
ln -s /usr/local/lib/python3.6/site-packages/cv2.so cv2.so

