# Modelos entrenados — acp-classifier

Artefactos de los modelos entrenados durante el trabajo de grado *Clasificación de péptidos
anticancerígenos mediante descriptores fisicoquímicos y modelos de lenguaje proteico*
(Escuela de Ingeniería de Sistemas e Informática, Universidad Industrial de Santander).

Este repositorio guarda lo que el paquete instalable no distribuye: **los clasificadores
entrenados de la Fase 2, las representaciones ya calculadas y los resultados crudos**. Existe
para que los experimentos del libro puedan reproducirse sin repetir la parte cara — generar los
*embeddings* requiere GPU; partiendo de estos archivos, no.

- Paquete instalable: <https://github.com/ACP-Classifier-test/acp-classifier>
- Instalación: `pip install acp-classifier`

## Qué hay aquí

### `modelos/` — 42 MB

Los siete clasificadores de la Fase 2 sobre los *embeddings* de ESM-2 8M (320 dimensiones).

Cada archivo es un `Pipeline` de scikit-learn que **incluye su propio `StandardScaler`**, ajustado
únicamente sobre `Training.fasta`. Es deliberado: el clasificador suelto, sin normalizar la
entrada, produce resultados sin sentido, y guardarlos juntos evita ese error.

### `representaciones/` — 90 MB

Vectores ya calculados, en formato `.npy`. Las filas siguen el orden de los archivos FASTA del
benchmark: 2.352 secuencias de entrenamiento (1.176 ACPs y 1.176 no-ACPs) y 3.370 de prueba
independiente (610 y 2.760).

| Directorio | Contenido | Dimensiones |
|---|---|---|
| `embeddings/` | ESM-2 8M, ESM-2 150M y ProtBERT, con sus etiquetas | 320 · 640 · 1.024 |
| `descriptores_577/` | Descriptores fisicoquímicos de la Fase 1 | 577 |
| `multimodal_897/` | Concatenación de descriptores y ESM-2 8M | 897 |

**Cuál usar.** Para ESM-2 8M hay dos variantes. Las que reproducen los resultados del libro son
`train_embeddings_esm2_8M_clean.npy` y `test_embeddings_esm2_8M_clean.npy`. La variante sin
`_clean` corresponde a una configuración anterior de *pooling* y da métricas distintas.

### `resultados/` — 5 MB

Los CSV de métricas de cada fase, los registros de ejecución y las figuras generadas.

## Verificación

`reentrenar.py` regenera los siete artefactos desde las representaciones y compara cada métrica
con la publicada. Esto es lo que sale:

| Clasificador | AUC-ROC obtenida | AUC-ROC publicada | F1 / Exactitud / MCC |
|---|---|---|---|
| Regresión Logística | **0,8745** | 0,8745 | coinciden exactamente |
| Random Forest | **0,8681** | 0,8681 | 0,6265 / 0,8326 / 0,5402 frente a 0,6213 / 0,8315 / 0,5330 |
| Extra Trees | **0,8670** | 0,8670 | 0,6122 / 0,8323 / 0,5200 frente a 0,6115 / 0,8341 / 0,5189 |
| SVM RBF | **0,8654** | 0,8654 | 0,6035 / 0,8300 / 0,5085 frente a 0,5958 / 0,8297 / 0,4985 |
| Gradient Boosting | 0,8603 | 0,8598 | coinciden en la tercera cifra |
| KNN | **0,8250** | 0,8250 | coinciden exactamente |
| MLP | 0,8181 | 0,8233 | difiere |

**La AUC-ROC —la métrica que el libro reporta como principal y con la que se seleccionó el modelo
final— se reproduce exactamente en cinco de los siete clasificadores**, incluida la Regresión
Logística, que es el modelo distribuido en el paquete y el que sostiene la cifra de 0,8745.

Las diferencias restantes son de tercera o cuarta cifra decimal y se concentran en las métricas
que dependen del umbral de 0,5 (F1, exactitud y MCC), en los modelos de conjunto y en el MLP. El
motivo es que el orden de construcción de los árboles, la calibración de probabilidades del SVC y
la inicialización de la red no quedan fijados solo por la semilla: dependen también del número de
hilos y de la versión de las bibliotecas. No afectan a ninguna conclusión del trabajo.

El detalle completo, métrica a métrica, está en `VERIFICACION.json`.

## Aviso sobre los artefactos anteriores

Los archivos que estaban en `codigo/results/models/` **no son estos y no deben publicarse como
los modelos del trabajo**. Los generaba `codigo/embeddings/classifier.py`, que construye su propia
partición estratificada sobre la unión de los dos archivos del benchmark — la vía de evaluación
ensayada y descartada que el libro documenta en la sección 6.1. Evaluados sobre el conjunto de
prueba oficial dan cifras infladas, de hasta 0,9449 de AUC-ROC, porque parte de ese conjunto
estuvo en su entrenamiento. Se conservan donde estaban, como registro de lo explorado, pero fuera
de este repositorio.

## Reproducibilidad

Los artefactos se serializaron con **scikit-learn 1.8.0** y **NumPy 2.4.1** sobre **Python 3.14**.
El número de versión de scikit-learn queda grabado dentro de cada archivo; con una versión
distinta, la biblioteca no garantiza que el objeto se reconstruya igual.

```python
import numpy as np, joblib
from sklearn.metrics import roc_auc_score

X = np.load("representaciones/embeddings/test_embeddings_esm2_8M_clean.npy")   # (3370, 320)
y = np.load("representaciones/embeddings/test_labels.npy")                     # (3370,)

modelo = joblib.load("modelos/esm2_8m_Logistic_Regression.pkl")   # incluye el scaler
print(roc_auc_score(y, modelo.predict_proba(X)[:, 1]))            # 0.8745
```

Para regenerarlo todo desde cero:

```bash
python reentrenar.py
```

Todas las particiones y búsquedas del trabajo emplean semilla fija (42).

## Integridad

`CHECKSUMS.sha256` lista la suma SHA-256 de los 58 archivos:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Datos de origen

El conjunto de referencia es el benchmark de mACPpred 2.0 (Sangaraju et al., 2024), distribuido
por sus autores. No se redistribuye aquí: estos archivos son representaciones derivadas de él.

## Licencia

MIT, la misma del paquete.
