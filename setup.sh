pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html


git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout cc87e7ec
pip install -e .
 
cd /content/I2T
pip install -r requirements_GRiT.txt
pip install -r requirements.txt