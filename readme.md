
##### en anaconda prompt xd
```powershell
conda activate env_sis421
cd "D:\proyecto_sis421_detector_trampas"
luego escribir:
D:
streamlit run app.py
```
---
si quieres reiniciar:
```powershell
conda deactivate
conda activate env_sis421
```
--- 

##### Para el entrenamiento del modelo de lstm, se uso el siguiente dataset, conjunto de datos de trampas en examenes, de la universidad de michigan:

```bash 
https://cvlab.cse.msu.edu/oep-dataset.html
```

para mejorar el modelo de retinanet se hizo una fusion de 2 datasets:

```bash 
https://universe.roboflow.com/eco-18oum/cellphone-ah9eu/dataset/7

https://universe.roboflow.com/trashdetection-bhjmn/paper-2jpbv/dataset/1
```