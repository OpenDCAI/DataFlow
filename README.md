# DataGym

This is the repo for PKU Data Centric ML

For environment setup, please using the following commands:

```
conda create -n datagym python=3.9
conda activate datagym
pip install -e .
```
<!-- pip install -r requirements.txt
pip install flash-attn==2.6.3
pip install pyiqa==0.1.12
pip install transformers==4.44.2
``` -->

If you want to evaluate text data only, using the following commands:
```
pip install -e .[text]
pip install flash-attn==2.6.3
python -m spacy download en_core_web_sm
```

For image data,
```
pip install -e .[image]
pip install pyiqa==0.1.12
pip install transformers==4.44.2
```

For video data,
```
pip install -e .[video]
```

When evaluating video-caption data, please run the following command to install modified CLIP for EMScore and PACScore:
```
pip install git+https://github.com/MOLYHECI/CLIP.git
```

All dependencies can be installed by:
```
pip install -e .[all]
pip install flash-attn==2.6.3
pip install pyiqa==0.1.12
pip install transformers==4.44.2
```


Please refer to test.py for config file usage. Use the following command to run with the config file

```
python test.py --config [your config file]
```
