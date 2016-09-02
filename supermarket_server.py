import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LREG

from flask import Flask, json, Response
import bokeh
from crossdomain import crossdomain

from string import Template
import regex as re

from google.protobuf import timestamp_pb2
from gcloud import storage

app = Flask(__name__)

DATASET_BUCKET = 'datasets-hf'

DEFAULT_CONTENT_TEMPLATE = Template("\n".join([
    "           <p>Predicci&oacute;n: ${pred}</p>",
    "           <p>Variables de mejora:</p>",
    "           <ul>",
    "               ${biom_l}",
    "           </ul>"
]))

## Utilidades para el manejo de diccionarios
# dict_find - Devuelve el indice de un elemento del diccionario, o None en caso de no encontrarlo
def dict_find(d, value):
    if isinstance(d, pd.DataFrame):
        d = d.to_dict()
    return safe_dict_get(dict(zip(d.values(), d.keys())), value)

# safe_dict_get - Devuelve el valor indicado para el diccionario seleccionado, o None en caso de no encontrarlo
def safe_dict_get(d, index):
    return d[index] if index in d.keys() else None


# Definimos la funcion de prognosis
def estimate_prognosis_weight_vector(cell, malignants, c_min=2, c_max=20, metric='euclidean'):
    # K-Means Cluster search on malignant cells
    silh = []
    km_model = [0] * (c_max - c_min + 1)
    for t in range(c_min, c_max):
        km_model[t - c_min] = KMeans(n_clusters=t).fit(malignants)
        silh.append(silhouette_score(malignants, km_model[t - c_min].labels_, metric=metric))

    idx = np.argmax(silh)
    n_clusters = idx + c_min
    cluster_lbl = km_model[idx].labels_
    # Cluster centroid retrieval and distance calculation
    distance, centroid = {}, {}
    for each in np.unique(cluster_lbl).tolist():
        centroid[each] = np.mean(malignants[cluster_lbl == each], 0)
        distance[each] = np.sqrt(np.sum(np.power(centroid[each] - cell, 2)))

    dist_v = {k: np.abs(centroid[k] - cell) for k, v in enumerate(distance.values())}[np.argmin(distance.values())]

    inv_dist = 1 / dist_v

    weights = inv_dist / np.sum(inv_dist)

    return weights


dataset_name = "Wholesale_customers_data"

# Descargamos el dataset de cancer del bucket de datasets
client = storage.Client()
for suffix in ['.csv', '-format.csv']:
    cblob = client.get_bucket(DATASET_BUCKET).get_blob(dataset_name + suffix)
    fp = open(dataset_name + suffix, 'wb')
    cblob.download_to_file(fp)
    fp.close()

# Cargamos los datos y aplicamos alguna transformacion
format = pd.read_csv(dataset_name + '-format.csv')

df = pd.read_csv(dataset_name + '.csv')
df['Class'] = (df['Delicatessen'] > 2000).astype(int)
df = df.drop('Delicatessen', axis=1)
labels = {'Usual Consumer': 0, 'Great Consumer': 1}

fmts = []
for c, d in {n: {format['field'][i]: float(vv) for i, vv in list(enumerate(v))}
             for n, v in dict(format.drop('field', axis=1).drop('Delicatessen', axis=1)).items()}.items():
    d.update(dict(title=c))
    fmts.append(d)

field_names = [field for field in df.drop(['Class'], 1)]
#flds = [dict(title=f, start=float(df[f].min()), end=float(df[f].max()), step=.5) for f in field_names]
#flds = [dict(zip(list(f.keys())+['value'], list(f.values())+[f['start']])) for f in flds]

# Separamos vectores de caracteristicas y etiquetas
X = np.array(df.drop(['Class'], 1), dtype=float)
y = np.array(df['Class'], dtype=float)

# Preparamos version normalizada para utilizar en los clasificadores
scalerX = StandardScaler()
scalery = StandardScaler()
Xn = np.apply_along_axis(scalerX.fit_transform,0,X)
yn = np.apply_along_axis(scalery.fit_transform,0,y)



# Definimos los clasificadores y los entrenamos
classifiers = {
    'SVC': SVC(probability=True).fit(Xn, y),
    'RF': RF().fit(Xn, y),
    'KNN': KNN().fit(Xn, y),
    'LREG': LREG().fit(Xn, y)
}

# Preparamos los vectores para la representacion 3D
#pca = PCA(n_components=3).fit(X)
pca = PCA(n_components=2).fit(X)
Xt = pca.transform(X)
xx1 = Xt[y == labels['Usual Consumer']][:, 0]
yy1 = Xt[y == labels['Usual Consumer']][:, 1]
#zz1 = Xt[y == labels['benign']][:, 2]

xx2 = Xt[y == labels['Great Consumer']][:, 0]
yy2 = Xt[y == labels['Great Consumer']][:, 1]
#zz2 = Xt[y == labels['malignant']][:, 2]

#base = dict(x=np.hstack((xx1, xx2)), y=np.hstack((yy1, yy2)), z=np.hstack((yy1, yy2)),
#            color=[1] * xx1.size + [3] * xx2.size)
base = dict(x=np.hstack((xx1, xx2)), y=np.hstack((yy1, yy2)), color=['blue'] * xx1.size + ['green'] * xx2.size)

@app.route('/predict/<chars>', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', methods=['POST', 'OPTIONS'], headers=None)
def predict(chars):

    chars = np.amin(X, axis=0) if re.match(r"[\[\(\{].*",chars) is None else json.loads(chars)
    aux = ""

    try:
        chars = np.apply_along_axis(scalerX.fit_transform,0,chars).tolist()

        clsf = {n: float(np.amax(f.predict_proba(chars))) for n, f in classifiers.items()}
        probs = dict(enumerate(clsf.values()))
        names = dict(enumerate(clsf.keys()))
        p_max_i = int(np.argmax(list(probs.values())))

        pred = "%s (%.2f%%, %s)" % (
            dict_find(labels, int(classifiers[names[p_max_i]].predict(chars).tolist()[0])),
            probs[p_max_i]*100,
            names[p_max_i]
        )

        prog_vec = estimate_prognosis_weight_vector(chars, Xn[y == labels['Great Consumer']])

        fnms = dict(enumerate(field_names))
        biom_l = "\n".join(["<li>%s: %.2f%%</li>" % (fnms[i], v*100) for i, v in enumerate(prog_vec)])

        aux = DEFAULT_CONTENT_TEMPLATE.substitute(**dict(pred=pred, biom_l=biom_l))

    except Exception as e:
        print("Error with input '%s': %s" % (chars, e))

    return json.jsonify({'results': aux})

@app.route('/compute/<chars>', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', methods=['POST', 'OPTIONS'], headers=None)
def compute(chars):

    chars = json.loads(chars)
    aux = {}

    try:

        chars = pca.transform(chars)[0]

        aux['x'] = np.hstack((base['x'], chars[0])).tolist()
        aux['y'] = np.hstack((base['y'], chars[1])).tolist()
        #aux['z'] = np.hstack((base['z'], chars[0])).tolist()
        aux['color'] = base['color'] + ['red']

    except Exception as e:
        print("Error with input '%s': %s" % (chars, e))

    return json.jsonify({'results': aux})

@app.route('/defaults', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', methods=['POST', 'OPTIONS'], headers=None)
def defaults():

    aux = {}

    try:

        chars = pca.transform(np.amin(X, axis=0))[0]

        aux['x'] = np.hstack((base['x'], chars[0])).tolist()
        aux['y'] = np.hstack((base['y'], chars[1])).tolist()
        #aux['z'] = np.hstack((base['z'], chars[0])).tolist()
        aux['color'] = base['color'] + ['red']# + ["rgba(0,255,0,1)"]

    except Exception as e:
        print("Error in 'defaults': %s" % (e,))

    return json.jsonify({'results': aux})

@app.route('/fields', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', methods=['POST', 'OPTIONS'], headers=None)
def fields():
    return json.jsonify({'results': fmts})

if __name__ == '__main__':
    app.run(port=50002)
