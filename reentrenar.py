# -*- coding: utf-8 -*-
"""
Reentrena los siete clasificadores de la Fase 2 con el protocolo adoptado en el
libro y guarda los artefactos en modelos/.

Protocolo: se parte de los embeddings de ESM-2 8M ya calculados, se normaliza con
StandardScaler ajustado SOLO sobre Training.fasta, se entrena con los
hiperparametros seleccionados por la validacion cruzada anidada y se evalua una
unica vez sobre Independent.fasta.

Cada modelo se guarda junto con su scaler dentro de un Pipeline, de modo que el
artefacto sea autosuficiente y no pueda usarse sin normalizar.
"""
import os, json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef

SEMILLA = 42
BASE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(BASE, 'representaciones', 'embeddings')
DEST = os.path.join(BASE, 'modelos')

# Hiperparametros seleccionados por la validacion cruzada anidada, y metricas
# publicadas en el libro sobre el conjunto de prueba independiente.
CONFIG = {
    'Logistic_Regression': (
        LogisticRegression(C=0.01, solver='saga', max_iter=5000, random_state=SEMILLA),
        (0.8745, 0.6230, 0.8362, 0.5341)),
    'Random_Forest': (
        RandomForestClassifier(max_depth=None, n_estimators=300, random_state=SEMILLA, n_jobs=-1),
        (0.8681, 0.6213, 0.8315, 0.5330)),
    'Extra_Trees': (
        ExtraTreesClassifier(max_depth=None, n_estimators=300, random_state=SEMILLA, n_jobs=-1),
        (0.8670, 0.6115, 0.8341, 0.5189)),
    'SVM_RBF': (
        SVC(C=10, gamma=0.001, probability=True, random_state=SEMILLA),
        (0.8654, 0.5958, 0.8297, 0.4985)),
    'Gradient_Boosting': (
        GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200,
                                   random_state=SEMILLA),
        (0.8598, 0.5988, 0.8234, 0.5031)),
    'KNN': (
        KNeighborsClassifier(metric='cosine', n_neighbors=11, weights='distance'),
        (0.8250, 0.5436, 0.7997, 0.4311)),
    'MLP': (
        MLPClassifier(alpha=0.01, hidden_layer_sizes=(256,), learning_rate_init=0.0001,
                      max_iter=1000, random_state=SEMILLA),
        (0.8233, 0.5298, 0.7819, 0.4133)),
}


def main():
    Xtr = np.load(os.path.join(REP, 'train_embeddings_esm2_8M_clean.npy'))
    Xte = np.load(os.path.join(REP, 'test_embeddings_esm2_8M_clean.npy'))
    ytr = np.load(os.path.join(REP, 'train_labels.npy'))
    yte = np.load(os.path.join(REP, 'test_labels.npy'))
    print('entrenamiento %s | prueba %s\n' % (Xtr.shape, Xte.shape))

    os.makedirs(DEST, exist_ok=True)
    informe = {}
    print('%-22s %-31s %s' % ('clasificador', 'obtenido (AUC/F1/Acc/MCC)', 'publicado'))
    for nombre, (clf, esperado) in CONFIG.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)]).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        obt = (roc_auc_score(yte, p), f1_score(yte, pred),
               accuracy_score(yte, pred), matthews_corrcoef(yte, pred))
        ok = all(abs(a - b) < 5e-4 for a, b in zip(obt, esperado))
        joblib.dump(pipe, os.path.join(DEST, 'esm2_8m_%s.pkl' % nombre))
        informe[nombre] = {'obtenido': [round(v, 4) for v in obt],
                           'publicado': list(esperado), 'coincide': ok}
        print('%-22s %-31s %s  %s' % (
            nombre,
            '%.4f / %.4f / %.4f / %.4f' % obt,
            '%.4f / %.4f / %.4f / %.4f' % esperado,
            'OK' if ok else '<<< NO COINCIDE'))

    with open(os.path.join(BASE, 'VERIFICACION.json'), 'w', encoding='utf-8') as fh:
        json.dump(informe, fh, indent=2, ensure_ascii=False)
    print('\ncoinciden %d de %d' % (sum(v['coincide'] for v in informe.values()), len(informe)))


if __name__ == '__main__':
    main()
