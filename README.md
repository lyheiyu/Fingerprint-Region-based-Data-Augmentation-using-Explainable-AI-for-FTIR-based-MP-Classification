# FRDA: Fingerprint Region based Data Augmentation using Explainable AI for FTIR based Microplastic Classification

## # FRDA
In this algorithm， there are 3 MP dataset, so you can find in 3 py file for different dataset augmenting.
The file (FTIR_DataAugmentation.py) is for the first Kedzierski’s dataset.

The file(FTIR_AugmentationMethodForseconddataset.py) is for the second Jung dataset.

The file (FTIR_augmentationMethodforThirddataset.py) is for the third Hannah dataset.

For save the model, you can use the FTIR_example.py to load the model from the SVM model for testing.

If you want to save the trained model, the codes is in the file(FTIR_AugmentationMethodForseconddataset.py).



## # Here is an example of saving and loading model.
from sklearn import svm  
from sklearn import datasets  
import joblib  

polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('D4_4_publication11.csv', 2, 1763)
x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3,
                                                            random_state=1)  
clf = svm.SVC()  
clf.fit(x_train, y_train)  
--saving model  
joblib.dump(clf, 'svm_model.pkl')  
--loading model
loaded_model = joblib.load('svm_model.pkl')  
--Testing  
predictions = loaded_model.predict(x_test)  
## # For XAI, You can uset the code FTIR_XCNNCLuster,py file
 

## # Datasets





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

[Software Research Institute](https://sri.ait.ie/) of [Technological University of the Shannon: Midlands Midwest](https://tus.ie/).
