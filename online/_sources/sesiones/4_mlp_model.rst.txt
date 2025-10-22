Modelar una serie de tiempo con MLP
===========================================

.. code:: ipython3

    import pandas as pd 
    # Leer el archivo CSV
    file_path = "/Users/nataliaacevedo/SeriesTemporalesDeepLearning/notebooks/Series con ML/global_weather_data_2015_2024.csv"
    data = pd.read_csv(file_path, encoding='latin1')
    usa_data = data[data['country'] == 'USA']
    new_york_data = usa_data[usa_data['city'] == 'New York']
    new_york_data.set_index('date', inplace=True)
    new_york_tavg = new_york_data['tavg']
    print(new_york_tavg.head())



.. parsed-literal::

    date
    2015-01-02 00:00:00    2.6
    2015-01-03 00:00:00    0.9
    2015-01-04 00:00:00    6.6
    2015-01-05 00:00:00    5.4
    2015-01-06 00:00:00   -6.4
    Name: tavg, dtype: float64


Preprocesamiento
^^^^^^^^^^^^^^^^
Extracción de características
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Crear las ventanas de tiempo para series temporales
    import numpy as np  
    
    def create_time_windows(series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series.iloc[i:i + window_size].values)  # Usar .iloc para acceder por posición
            y.append(series.iloc[i + window_size])          # Usar .iloc para el valor objetivo
        return np.array(X), np.array(y)
    
    # Crear ventanas de 7 días
    X, y = create_time_windows(new_york_tavg, 7)
    
    X = np.array(X)
    y = np.array(y).flatten()  # Asegúrate de que y sea 1D
    
    print("Ventanas de entrada (X):")
    print(X.shape)
    print(X)
    print("Valores objetivo (y):")
    print(y.shape)
    print(y)



.. parsed-literal::

    Ventanas de entrada (X):
    (3642, 7)
    [[  2.6   0.9   6.6 ...  -6.4  -7.1 -11.1]
     [  0.9   6.6   5.4 ...  -7.1 -11.1  -5.1]
     [  6.6   5.4  -6.4 ... -11.1  -5.1  -6.6]
     ...
     [  6.8   2.4  -0.3 ...  -7.7  -0.4   1.1]
     [  2.4  -0.3  -6.6 ...  -0.4   1.1  -1.3]
     [ -0.3  -6.6  -7.7 ...   1.1  -1.3  -0.6]]
    Valores objetivo (y):
    (3642,)
    [-5.1 -6.6 -5.1 ... -1.3 -0.6  5.1]


**Ventana de entrada (X):**

Cada ventana contiene window_size valores consecutivos de la serie.

Por ejemplo, si ``window_size = 3`` y la serie es [1, 2, 3, 4, 5], las
ventanas serán:

``X[0] = [1, 2, 3]``

``X[1] = [2, 3, 4]``

**Valor objetivo (y):**

El valor objetivo es el siguiente valor en la serie después de la
ventana. En el ejemplo anterior: ``y[0] = 4`` (el valor después de [1,
2, 3]) ``y[1] = 5`` (el valor después de [2, 3, 4])

**Sin fuga de datos:**

No se utiliza información del futuro para construir las ventanas de
entrada (X).

El valor objetivo (y) siempre está fuera de la ventana de entrada y
corresponde a un punto en el futuro. Consideraciones adicionales:

**Orden temporal:** Asegúrate de que la serie esté correctamente
ordenada por tiempo antes de aplicar esta función.

**Normalización:** Si planeas normalizar los datos, hazlo después de
dividir los datos en ventanas para evitar fuga de información entre las
ventanas.

Escalar o Normalizar la serie
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recomendación práctica para redes neuronales:**

Para series financieras o ruidosas → RobustScaler o StandardScaler.

Para series con crecimiento o escala amplia → Log + MinMaxScaler.

Para LSTM/GRU/CNN → mantener datos en rango [−1, 1] mejora la
convergencia.

+--------------------+----------+----------+------------------+-------------------+
| Método             | Robustez | Mantiene | Ideal para       | Precauciones      |
|                    | a        | orden    |                  |                   |
|                    | outliers | temporal |                  |                   |
+====================+==========+==========+==================+===================+
| **Min–Max**        | ❌       | ✅       | Datos acotados,  | Sensible a        |
|                    |          |          | sin picos        | extremos          |
+--------------------+----------+----------+------------------+-------------------+
| **StandardScaler** | ⚠️       | ✅       | Series estables  | Puede amplificar  |
|                    |          |          |                  | outliers          |
+--------------------+----------+----------+------------------+-------------------+
| **RobustScaler**   | ✅       | ✅       | Series           | Reduce            |
|                    |          |          | financieras o    | sensibilidad a    |
|                    |          |          | ruidosas         | cambios suaves    |
+--------------------+----------+----------+------------------+-------------------+
| **Log Transform**  | ✅       | ✅       | Datos positivos, | No usar con       |
|                    |          |          | crecimiento      | valores negativos |
|                    |          |          | exponencial      |                   |
+--------------------+----------+----------+------------------+-------------------+
| **Power            | ✅       | ✅       | Distribuciones   | Requiere ajuste   |
| Transform**        |          |          | sesgadas         | cuidadoso         |
+--------------------+----------+----------+------------------+-------------------+
| **Incremental      | ✅       | ✅       | Series en        | Precisión menor   |
| Scaling**          |          |          | streaming        | si cambian las    |
|                    |          |          |                  | estadísticas      |
+--------------------+----------+----------+------------------+-------------------+
| **Per-variable     | ✅       | ✅       | Series           | Debe aplicarse    |
| Scaling**          |          |          | multivariadas    | coherentemente al |
|                    |          |          |                  | predecir          |
+--------------------+----------+----------+------------------+-------------------+

.. code:: ipython3

    from sklearn.preprocessing import MinMaxScaler
    
    # Crear instancias de MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Ajustar y transformar X
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Ajustar y transformar y
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    print("X escalado:")
    print(X_scaled)
    print("y escalado:")
    print(y_scaled)


.. parsed-literal::

    X escalado:
    [[0.34051724 0.30387931 0.42672414 ... 0.14655172 0.13146552 0.04525862]
     [0.30387931 0.42672414 0.40086207 ... 0.13146552 0.04525862 0.17456897]
     [0.42672414 0.40086207 0.14655172 ... 0.04525862 0.17456897 0.14224138]
     ...
     [0.43103448 0.3362069  0.27801724 ... 0.11853448 0.27586207 0.30818966]
     [0.3362069  0.27801724 0.14224138 ... 0.27586207 0.30818966 0.25646552]
     [0.27801724 0.14224138 0.11853448 ... 0.30818966 0.25646552 0.27155172]]
    y escalado:
    [0.17456897 0.14224138 0.17456897 ... 0.25646552 0.27155172 0.39439655]


.. code:: ipython3

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    
    # Crear el modelo MLP
    model = Sequential([
        Input(shape=(X.shape[1],)),  # Define explícitamente la forma de entrada   
        Dense(32, activation='relu'),   # Capa densa con 64 neuronas
        Dense(1)  # Capa de salida con 1 neurona (predicción para 1 día)
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

.. code:: ipython3

    # Entrenar el modelo
    history = model.fit(X_scaled, y_scaled, epochs=500, batch_size=32, validation_split=0.2, verbose=1)


.. parsed-literal::

    Epoch 1/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0584 - mse: 0.0584 - val_loss: 0.0085 - val_mse: 0.0085
    Epoch 2/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 0.0090 - mse: 0.0090 - val_loss: 0.0066 - val_mse: 0.0066
    Epoch 3/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0071 - mse: 0.0071 - val_loss: 0.0054 - val_mse: 0.0054
    Epoch 4/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 823us/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.0048 - val_mse: 0.0048
    Epoch 5/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 806us/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.0044 - val_mse: 0.0044
    Epoch 6/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0053 - mse: 0.0053 - val_loss: 0.0041 - val_mse: 0.0041
    Epoch 7/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 801us/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 8/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 9/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 10/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 11/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 12/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 13/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 14/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 15/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 16/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 17/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 951us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 18/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 19/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 20/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 21/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 22/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 23/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 24/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 25/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 26/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 27/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 28/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 29/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 30/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 31/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 32/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 33/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 34/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 35/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 36/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 37/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 38/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 39/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 40/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 41/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 42/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 878us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 43/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 44/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 45/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 46/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 855us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 47/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 48/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 853us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 49/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 841us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 50/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 831us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0042 - val_mse: 0.0042
    Epoch 51/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 844us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 52/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 824us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 53/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 54/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 824us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 55/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 828us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 56/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 828us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 57/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 816us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 58/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 796us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 59/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 836us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 60/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 815us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 61/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 866us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 62/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 63/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 64/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 65/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 66/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0047 - val_mse: 0.0047
    Epoch 67/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 68/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 69/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 70/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 71/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 72/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 73/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 74/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 75/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 76/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 77/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 78/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 79/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 80/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 81/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 82/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 826us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 83/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 84/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 85/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0046 - val_mse: 0.0046
    Epoch 86/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0045 - val_mse: 0.0045
    Epoch 87/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 88/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 89/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 90/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 91/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 92/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 874us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 93/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 806us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 94/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 807us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 95/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 830us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 96/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 811us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 97/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 808us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 98/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 840us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 99/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 826us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 100/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 818us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 101/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 102/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 103/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 104/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 105/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 818us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 106/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 107/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 108/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 109/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 110/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 111/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 112/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 801us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 113/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0041 - val_mse: 0.0041
    Epoch 114/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 115/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 116/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 117/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 118/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 119/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 120/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 121/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 122/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 123/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 124/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 125/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 126/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 127/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0048 - val_mse: 0.0048
    Epoch 128/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 129/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 809us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 130/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 131/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 132/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 133/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 134/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 135/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 136/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 137/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 138/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 139/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 140/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 141/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 142/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 143/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 144/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 145/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 146/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 796us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 147/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 148/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 149/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 150/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 151/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0044 - val_mse: 0.0044
    Epoch 152/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 153/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 154/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 155/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 797us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 156/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 817us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 157/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 158/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 159/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 160/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 161/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 162/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 163/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 164/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 165/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 166/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 167/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 168/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 169/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 170/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 171/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 172/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 173/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0041 - val_mse: 0.0041
    Epoch 174/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 175/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 176/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 177/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 178/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 179/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 180/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 181/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 182/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 797us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 183/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0044 - val_mse: 0.0044
    Epoch 184/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 185/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 186/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 187/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 188/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0041 - val_mse: 0.0041
    Epoch 189/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 190/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 191/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 192/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 193/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 194/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 195/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 196/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 197/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 198/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 199/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 200/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 201/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 202/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 203/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 818us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 204/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 205/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 206/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 207/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 208/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 209/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 210/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 211/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 212/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 213/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 214/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 215/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 216/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 217/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 218/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 219/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 220/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 221/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 222/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 223/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 224/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 225/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 226/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 227/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 228/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 229/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 230/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 231/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 232/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 233/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 234/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 235/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 236/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0045 - val_mse: 0.0045
    Epoch 237/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 238/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 239/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 240/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 241/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 242/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 243/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0043 - val_mse: 0.0043
    Epoch 244/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 245/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 246/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 247/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 248/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 249/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 250/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 251/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 252/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 807us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 253/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 254/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 255/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 256/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 257/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 258/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 259/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 260/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 261/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 262/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 263/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 264/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 265/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 266/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 267/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 268/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 269/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 270/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 271/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 272/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 273/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 828us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 274/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 275/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 276/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 277/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 278/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 279/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 280/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 281/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 282/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 283/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 284/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 285/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 286/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 287/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 288/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 289/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 290/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 291/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 292/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 293/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 294/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 295/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 296/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 297/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 298/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 798us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 299/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 300/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 301/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 302/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 805us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 303/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 304/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 305/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 306/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 938us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 307/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 308/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 309/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 767us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 310/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 311/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 312/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 313/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 314/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 315/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 316/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 815us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 317/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 318/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 319/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 320/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 321/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 322/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 323/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 324/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 325/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 326/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 327/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 328/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 329/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 330/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 331/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 332/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 333/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 334/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 335/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 336/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 337/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 338/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 818us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 339/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 340/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 341/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 342/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 343/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 344/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 345/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 346/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 347/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 348/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 349/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 801us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 350/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0045 - val_mse: 0.0045
    Epoch 351/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 352/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 353/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 354/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 355/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 356/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 357/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 358/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 359/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 360/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 361/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 362/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 363/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 364/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 365/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 366/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 367/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 368/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 369/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 370/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 371/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 372/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 373/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 374/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 808us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 375/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 376/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 377/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 378/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 379/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 380/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 381/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 382/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0042 - val_mse: 0.0042
    Epoch 383/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 384/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 385/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 386/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 387/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 787us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 388/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 389/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 390/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 391/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 392/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 804us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 393/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 394/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 395/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 396/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 397/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 398/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 399/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 400/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 401/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 402/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 403/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 404/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 405/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 406/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 407/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 408/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 409/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 847us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 410/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 411/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 412/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 413/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 414/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 415/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 416/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 417/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 418/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 419/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 420/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 421/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 422/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 423/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 424/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 425/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 426/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 828us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 427/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 803us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 428/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 429/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 757us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 430/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 431/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 432/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 433/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 434/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 435/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 436/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 437/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 438/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 439/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 440/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 441/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 442/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 805us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 443/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 444/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 445/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 446/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 447/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 448/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 449/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 450/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0040 - val_mse: 0.0040
    Epoch 451/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 452/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 453/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 454/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 455/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 456/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 457/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 458/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 459/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 809us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 460/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 803us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 461/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 462/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 463/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 464/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 465/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 466/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 467/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 468/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 469/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 470/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0045 - val_mse: 0.0045
    Epoch 471/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 472/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 473/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0039 - val_mse: 0.0039
    Epoch 474/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 475/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 476/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 477/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 478/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 479/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 480/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 481/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 482/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 483/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036
    Epoch 484/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0038 - val_mse: 0.0038
    Epoch 485/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 486/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 487/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 488/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 489/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 837us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 490/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 491/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0037 - val_mse: 0.0037
    Epoch 492/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 840us/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 493/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 494/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 495/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 496/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 497/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 498/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0034 - val_mse: 0.0034
    Epoch 499/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0035 - val_mse: 0.0035
    Epoch 500/500
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0036 - val_mse: 0.0036


.. code:: ipython3

    import matplotlib.pyplot as plt
    
    # Extraer las pérdidas de entrenamiento y validación
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Pérdida de entrenamiento')
    plt.plot(val_loss, label='Pérdida de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de entrenamiento y validación')
    plt.legend()
    plt.grid()
    plt.show()



.. image:: mlp_model_files/mlp_model_9_0.png


.. code:: ipython3

    # Realizar predicciones
    y_pred = model.predict(X_scaled)
    # Invertir la escala de las predicciones
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    



.. parsed-literal::

    [1m114/114[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 498us/step


.. code:: ipython3

    # Graficar valores reales vs predichos
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='Valores reales', alpha=0.7)
    plt.plot(y_pred_rescaled, label='Valores predichos', alpha=0.7)
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.title('Valores reales vs predichos')
    plt.legend()
    plt.grid()
    plt.show()



.. image:: mlp_model_files/mlp_model_11_0.png


.. code:: ipython3

    # Extraer los residuos
    residuals = y - y_pred_rescaled.flatten()
    print(residuals.shape)


.. parsed-literal::

    (3642,)


.. code:: ipython3

    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuales', alpha=0.7, linestyle='', marker='o')  # Cambiar líneas por puntos
    plt.xlabel('Índice')
    plt.ylabel('Valor residual')
    plt.title('Residuales de la predicción')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid()
    plt.show()



.. image:: mlp_model_files/mlp_model_13_0.png



Optimización de Hiperparámetros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------+---------------------------------------+
| Escenario                    | Recomendado                           |
+==============================+=======================================+
| Modelos clásicos (SVM,       | ``GridSearchCV``,                     |
| Random Forest, XGBoost)      | ``RandomizedSearchCV``, ``Optuna``,   |
|                              | ``HyperOpt``                          |
+------------------------------+---------------------------------------+
| Modelos de Deep Learning     | ``Keras Tuner``, ``Optuna``,          |
| (Keras/TensorFlow)           | ``Ray Tune``                          |
+------------------------------+---------------------------------------+
| Búsqueda escalable o         | ``Ray Tune``, ``BOHB``,               |
| distribuida                  | ``Optuna + Ray``                      |
+------------------------------+---------------------------------------+
| AutoML / pipelines           | ``TPOT``, ``Auto-Sklearn``,           |
| automáticos                  | ``AutoKeras``                         |
+------------------------------+---------------------------------------+
| Exploración evolutiva o      | ``Nevergrad``, ``DEAP``, ``PyGAD``    |
| heurística                   |                                       |
+------------------------------+---------------------------------------+

.. code:: ipython3

    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.base import BaseEstimator, RegressorMixin
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam, SGD
    import numpy as np
    
    # Crear un wrapper personalizado para Keras
    class KerasRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, optimizer='adam', neurons=64, batch_size=32, epochs=10, verbose=0):
            self.optimizer = optimizer
            self.neurons = neurons
            self.batch_size = batch_size
            self.epochs = epochs
            self.verbose = verbose
            self.model = None
    
        def build_model(self):
            model = Sequential([
                Dense(self.neurons, activation='relu', input_shape=(X.shape[1],)),
                Dense(self.neurons // 2, activation='relu'),
                Dense(1)
            ])
            optimizer = Adam() if self.optimizer == 'adam' else SGD()
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            return model
    
        def fit(self, X, y):
            self.model = self.build_model()
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
            return self
    
        def predict(self, X):
            return self.model.predict(X, verbose=0).flatten()  # Asegúrate de devolver un array 1D
    
    # Crear el modelo envuelto para scikit-learn
    model = KerasRegressor()
    print("wrapper creado")
    


.. code:: ipython3

    # Definir el diccionario de hiperparámetros a ajustar
    param_grid = {
        'optimizer': ['adam', 'sgd'],
        'neurons': [32, 64],
        'batch_size': [16],
        'epochs': [10]
    }
    


.. code:: ipython3

    # Configurar la validación cruzada para series temporales
    cv = TimeSeriesSplit(n_splits=3)
    
    # Configurar GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=0)
    
    # Ajustar el modelo
    grid_result = grid.fit(X, y)
    
    # Mostrar los mejores parámetros y el mejor puntaje
    print(f"Mejores parámetros: {grid_result.best_params_}")
    print(f"Mejor puntaje (MSE): {-grid_result.best_score_}")
