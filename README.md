# deepfake-hand-biomarker
PLEASE NOTE: This was developed for a bachelor thesis. So this is in experimental status. 
- Keras-based binary classifier (fake/real) using hand extracts to spot diffusion-generated deepfake images
- dataset:
  - 500 diffusion-generated hands (Midjourney Diffusion Model)
  - 500 real hands
- includes "from-scratch" training and transfer learning training (VGG-16):
  - Overfitting prevention:
    - dropout-layer (from-scratch model)
    - early stopping (val loss monitoring)
  - VGG-16 static (non-trainable layer)

## Requirements/Libraries
- Jupyter Notebook
- Anaconda Navigator 
- Keras/Tensorflow
- Matplotlib
- Scikit-Learn
- Pandas
### Hand Extractor for extracting hands
- make sure to use Python 3.8
- mediapipe (for hand detection)
- numpy
- math
- cv2
- os
  
