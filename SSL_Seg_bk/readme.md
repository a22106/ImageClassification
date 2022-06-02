# create new conda environment
conda create -n (env-name) python=3.9

# intall modules
### pytorch
- CPU version
!conda install pytorch torchvision torchaudio cpuonly -c pytorch

- CUDA 11.1
!pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

- CUDA 11.3
!conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

- CUDA 11.6
!pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116

### utils
- pandas, opencv, imgaug, albumentations
pip install pandas
pip install numpy
pip install opencv-python

- imgaug
conda config --add channels conda-forge
conda install imgaug

- albumentations
pip install -U albumentations