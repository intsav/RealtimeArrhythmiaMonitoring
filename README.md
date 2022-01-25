# ECG Real-time Arrhythmia Monitoring
Code files to accompany the paper *"ECG-based Real-time Arrhythmia Monitoring Using Quantized Deep Neural Networks: A Feasibility Study".*
For more details about this study, please visit: https://intsav.github.io/realtime_ecg.html


### Step 1 - Requirements
> Clone this repository
> >		git clone git@github.com:intsav/RealtimeArrhythmiaMonitoring

> Install virtualenv
> >		pip3 install virtualenv

> Create and activate Python 3.7 environment
> >		virtualenv -p python3.7

> Install requirements
> >		./setup.sh


### Step 2 - Data
> In the root directory create a new directories called **model** and **data**
> >		mkdir {models,data}

> Fetch and save dataset
> >     wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KIBxRB12tbEop02Dj_sLBuZvPgu3ua6e' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KIBxRB12tbEop02Dj_sLBuZvPgu3ua6e" -O data/mitdb_360_train.csv  && rm -rf /tmp/cookies.txt
> >		wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1epF6BHCrTUOrpILBUp4xg160guVy_Jsr' -O data/mitdb_360_test.csv


### Step 3 - Training
> Run the following command from the root directory
> >     python3 code/train.py
At each epoch the model is saved in **models** directory.


### Step 4 - Test baseline model
> Run the following command from the root directory by selecting your best model
> >     python3 code/test.py --model models/**FILENAME**


### Step 5 - Test quantized model
> Download quantized weights
> >		wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pY--6B4xNpcEMixEEwVgoD1h5AVvk2pW' -O models/ecg_quant.tflite

> Test quantized model
> > 	python3 code/test_quant.py
