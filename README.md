# Online Parameter-Free Fine Tuning for Heterogeneous Sequences of Tasks
This repository contains the source code for the thesis submitted in part requirement for the MSc Computational Statistics and Machine Learning at University College London.


### 🐳 Installation using Docker 

📍The following steps should be conducted on your terminal and make sure you have already installed Docker on your machine

1. `git clone` this repository to your local machine 🖥
2. `cd` to the cloned repository 
3. run `docker build -t paramfreefinetuning . ` to build the docker image, here you can change the image name `paramfreefinetuning` to whatever you want to call it
4. run `docker run -it -p 8888:8888 paramfreefinetuning` to build the Docker container through the image, can the terminal will show a new line like `root@xxxxxxxxxxxx:/project# ` which means you've successfully entered the container you just built! 
5. run `jupyter lab --allow-root --port=8888 --ip=0.0.0.0` in the container (in other words, type the command after `/project#`) and it will give you the link to access Jupyter Lab
6. copy and paste the provided link on the web to launch the Jupyter Lab and then you are good to go!

### 🛠 Further Documentation for the code under construction 

Stay tuned! 
