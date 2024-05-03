# deepfake-hand-biomarker
- Keras-based binary classifier (fake/real) using hand extracts to spot diffusion-generated deepfake images
- dataset:
  - 500 diffusion-generated hands (Midjourney Diffusion Model)
  - 500 real hands
- includes "from-scratch" training and transfer learning training (VGG-16):
  - Overfitting prevention:
    - dropout-layer (from-scratch model)
    - early stopping (val loss monitoring)
  - VGG-16 static (non-trainable layer)

## Requirements
- Jupyter Notebook
- Anaconda Navigator 
- Keras/Tensorflow
- Matplotlib
- Scikit-Learn
- Pandas
  
