# Installation instructions

## Without docker:

### Quick Windows 10 install instructions:
- [x] Nvidia GPU Support
- [ ] AMD GPU Support
- [ ] CPU Support
1. Download and install python3 for Microsoft Windows: https://www.python.org/downloads/windows/
2. Install Cuda 10, NOT 10.1.: https://developer.nvidia.com/cuda-downloads (All you should need is the runtime)
3. Install Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
4. Download a .zip of the gpt-2 software: https://github.com/openai/gpt-2/archive/master.zip
5. Open the zip, copy the software to an easily accessible folder i.e. C:\Users\yourusername\Documents\gpt-2-master
6. Open PowerShell: Start menu - > right-click powershell (or command prompt) -> Run as Administrator
7. Run this command to install tensorflow-gpu:
   - ```python -m pip install --user tensorflow-gpu```
8. Run this command to install requirements.txt after moving into the software's directory```
   - ```cd C:\Users\YOURUSERNAME\Documents\gpt-2-master```
   - ```python -m pip install --user -r requirements.txt```
9. Download the model of your choice:
   - ```python download_model.py 124M```
   - ```python download_model.py 355M``` 
   - ```python download_model.py 774M```
10. Install Cuda: https://developer.nvidia.com/cuda-downloads
11. Run the interactive sample, I reccomend making a backup of the models directory:
   - ```xcopy C:\Users\YOURUSERNAME\Documents\gpt-2-master\models C:\Users\YOURUSERNAME\Documents\gpt-2-master\modelsbackup /O /X /E /H /K```
   - ```python src/interactive_conditional_samples.py```
10. Adjust settings as needed.

### Quick Ubuntu 18.04 install instructions:
- [x] Nvidia GPU Support
- [ ] AMD GPU Support
- [x] CPU Support
1. Update apt:
    - ```apt update```
2. Install python3, python3-pip and git:
    - ```apt install python3 python3-pip git```
3. Install cuda if you are running on a GPU: https://developer.nvidia.com/cuda-downloads
4. Clone the software and navigate into the folder:
    - ```git clone https://github.com/openai/gpt-2.git```
    - ```cd gpt-2```
5. Install tensorflow gpu or cpu: (package tensorflow for cpu or tensorflow-gpu for gpu assuming you have cuda and drivers)
already installed.):
    - ```python3 -m pip install tensorflow```
    - ```python3 -m pip install tensorflow-gpu```
6. Install the requirements file:
    - ```python3 -m pip install -r requirements.txt```
7. Download a model: (The current largest model is 774M with a capital M):
   - ```python3 download_model.py 124M```
   - ```python3 download_model.py 355M``` 
   - ```python3 download_model.py 774M```

8. Run the interactive sample, I reccomend making a backup of the models directory:
    - ```cp models modelsbackup -R```
    - ```python3 src/interactive_conditional_samples.py```
9. Adjust settings as needed.
