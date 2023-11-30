# FRDA: Fingerprint Region based Data Augmentation using Explainable AI for FTIR based Microplastic Classification





## Introduction




The repo provide 2 tools:  

- FRDA, a data augmentation tool for FTIR samples

- 1D-CAM, a Explainable AI tool for analysing FTIR samples.

Here is the introduction of each code file.

FTIR_1DCAM.py is used for training the 1DCAM model that is built by Keras. You can use it to train you own model,
FTIR_AugmentationBased....py is used for ESMA methods. 
FTIR_GaussinMixture.py is used for clustering each influence curve obtained from the 1DCAM and returning the separate point.FTIR_XCNN is the identical file to the FTIR_1DCAM.
FTIR_fit_least_square.py is used for separating the curve and randomly selecting and combining the data.
PLS.py is used for baseline reducing algorithms.utils.py is used for reading data and plotting the confusion matrix.
utils.py is used for reading data and plotting the confusion matrix.



## FRDA Usage




This repo provides 3 datasets:

- Kedzierski’s

- Jung's

- Hannah's




The FRDA tool performs data augmentation for each dataset and then train and validate different ML models


Kedzierski’s dataset with FRDA




```

python FTIR_DataAugmentation_Kedzierski.py

```

Jung’s dataset with FRDA




```

python FTIR_DataAugmentation_Jung.py

```

Jung’s dataset with FRDA




```

python FTIR_DataAugmentation_Hannah.py

```

## 1DCAM Usage

The 1D-CAM can be used as follows: 
-	Training a classification model using 1D-CNN layer
-	Creating 1D-CAM based the trained network (using the layer before the output layer and parameters).
-	Inputting the new sample data to classification model and obtaining classification result.
- Inputting the identical data into 1DCAM and using the classification result to calculate the influence curve. 


Generating influence data curve using 1DCAM 
```

python FTIR_1DCAM.py

```

## # Citation

Use the below bibtex to cite us.

```BibTeX
@article{yan2023 FRDA,
  title={FRDA: Fingerprint Region based Data Augmentation using Explainable AI for FTIR based Microplastic Classification},
  author={Yan, Xinyu and Cao, Zhi and Murphy, Alan and Ye, Yuhang and Wang, Xinwu and Qiao, Yuansong},
  journal={},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={Elsevier}
}

@misc{yan2023ensemble,
  title={FRDA: Fingerprint Region based Data Augmentation using Explainable AI for FTIR based Microplastic Classification},
  author={Yan, Xinyu and Cao, Zhi and Murphy, Alan and Ye, Yuhang and Wang, Xinwu and Qiao, Yuansong},
  year={2023},
  publisher={Github},
  howpublished={\url{https://github.com/lyheiyu/Fingerprint-Region-based-Data-Augmentation-using-Explainable-AI-for-FTIR-based-MP-Classification/}},
}

```
* * * * *

## Developed by

[Software Research Institute](https://sri-tus.eu/) of [Technological University of the Shannon: Midlands Midwest](https://tus.ie/).
