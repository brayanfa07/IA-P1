import tensorflow as tf
import pandas as pd
import numpy as np

# Variables y constantes
n_classes = 2
tmp_dir = "tmp/model"
steps_spec = 2000
epochs_spec = 15
hidden_units_spec = [10,20,10]

# Data set file csv from TECDigital
file_name = "cancer.csv"

# Conjunto con el nombre de las caracteristicas
features = ['radius','texture','perimeter']

# Esta es la etiqueta que queremos predecir
labels = ['diagnosis_numeric']

# De donde se cargaran los datos
data_url = 'https://drive.google.com/open?id=1F15vKW9ZjnzunU-O4TxmtPH6mNaPZMmF'

# Se cargan los datos
file_name = tf.keras.utils.get_file('wsbc.csv', data_url)
data_set = pd.read_csv(file_name, delimiter=',')

# Primero de desordena todo el conjunto de datos
flag_shuffle = True

if flag_shuffle:
    randomized_data = data_set.reindex(np.random.permutation(data_set.index))
else:
    randomized_data = data_set

# Ahora se saca cuantos son los datos que se necesitan para entrenamiento y pruebas
total_records = len(randomized_data)
training_set_size = int(total_records * 0.8)
test_set_size = total_records - training_set_size


# Obtendo los diferentes conjuntos para entrenamiento, tanto de caracteristicas como de etiquetas. 
training_features = randomized_data.head(training_set_size)[features].copy()
training_labels = randomized_data.head(training_set_size)[labels].copy()

# Obtendo los diferentes conjuntos para pruebas, tanto de caracteristicas como de etiquetas. 
testing_features = randomized_data.tail(test_set_size)[features].copy()
testing_labels = randomized_data.tail(test_set_size)[labels].copy()

# A Tensor Flow hay que especificarle como van a ser los datos de las caracteristicas que va a tener.
feature_columns = [tf.feature_column.numeric_column(key) for key in features]

# Se crea la red neuronal que clasifica los datos
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, 
    hidden_units=hidden_units_spec, 
    n_classes=n_classes, 
    model_dir=tmp_dir)

# Se crea la funcion de entrada para entrenamiento
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)

# Se realiza el entrenamiento de la red con la funcion de entrada
classifier.train(
    input_fn=lambda:train_input_fn(training_features, training_labels, epochs_spec),
    steps=steps_spec)

# Se crea la funcion de entrada para pruebas
def test_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.batch(batch_size)

# Se evalua la precision (accuracy) *100
accuracy_score = classifier.evaluate(input_fn=lambda:test_input_fn(testing_features, testing_labels, epochs_spec))
print("Precision = " + str(classifier.evaluate(input_fn=lambda:train_input_fn(training_features, training_labels, epochs_spec),
    steps=steps_spec)))
print("Precision = " + str(accuracy_score))
