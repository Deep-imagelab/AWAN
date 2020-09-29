# Download data
[NTIRE 2020 Spectral Reconstruction Challenge - Track 2: Real World](https://competitions.codalab.org/competitions/22226).

- Place the data at *NTIRE2020* and arrange the directories as follows:

    *NTIRE2020/NTIRE2020_Train_Spectral*  
    --ARAD_HS_0001.mat  
    --ARAD_HS_0002.mat  
    ......  
    --ARAD_HS_0450.mat  
    
    *NTIRE2020/NTIRE2020_Train_RealWorld*  
    --ARAD_HS_0001_RealWorld.jpg  
    --ARAD_HS_0002_RealWorld.jpg  
    ......  
    --ARAD_HS_0450_RealWorld.jpg  
    
    *NTIRE2020/NTIRE2020_Validation_Spectral*  
    --ARAD_HS_0451.mat  
    --ARAD_HS_0453.mat  
    ......  
    --ARAD_HS_0465.mat  
    
    *NTIRE2020/NTIRE2020_Validation_RealWorld*  
    --ARAD_HS_0451_RealWorld.jpg  
    --ARAD_HS_0453_RealWorld.jpg  
    ......  
    --ARAD_HS_0465_RealWorld.jpg  
    
# Train the model
- If you are running the code for the first time, remember running data pre-processing firstly.

    python train_data_preprocess.py  
    python valid_data_preprocess.py  
    
- Otherwise, run training directly.

    python main.py  
    

