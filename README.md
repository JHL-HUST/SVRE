## Stochastic Variance Reduced Ensemble (SVRE)
This repository contains code to reproduce results from the paper:
**Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**(CVPR2022)
We provide an example of the SVRE method, and the complete experimental code and data will be released soon. 



## Datesets And models
To run the code, you should download pretrained models and the data. 

Please place [pre-trained models](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw) under the models/ directory.  

Please unzip the data and place the [data](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set) under the dataset/ directory.


## Rrquirements
- Python >= 3.6.5

- Tensorflow-gpu >= 1.14.0 

- Numpy >= 1.15.4 

- opencv >= 3.4.2

- scipy >= 1.1.0

- pandas >= 1.0.1

- imageio >= 2.6.1

  

## File Description
- `SVRE-I-FGSM.py`,`Ens-I-FGSM.py`,`SVRE-MI-FGSM.py`,`Ens-MI-FGSM.py` :  Generate adversarial examples.

- `eval.py`: Eval the efficacy of attack method.

- `./models`: Pre-trained models.

- `./nets`:  Code for model architecture.

- `./dataset`: The images used in the experiment and their label information. 

  


## Experiments
We provide an example of generating the adversarial examples on the ensemble of four normally trained models, ie. Inc-v3, Inc-v4, Res-15 and IncRes-v2, and test the transferability of the crafted adversaries on defense models.

To generate adversarial exmples of SVRE-I-FGSM and Ens-I-FGSM:
```
CUDA_VISIBLE_DEVICES=[gpu id] python SVRE-I-FGSM.py
CUDA_VISIBLE_DEVICES=[gpu id] python Ens-I-FGSM.py
```

To eval the efficacy of SVRE-I-FGSM and Ens-I-FGSM:
```
CUDA_VISIBLE_DEVICES=[gpu id] python eval.py  --eval_file ./results/SVRE-I-FGSM/
CUDA_VISIBLE_DEVICES=[gpu id] python eval.py  --eval_file ./results/Ens-I-FGSM/
```



## Acknowledgements

In order to ensure that our personal information is not leaked, we obtain the download link of the model from open source repositories, eg. [SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM). We thank the authors for sharing.
