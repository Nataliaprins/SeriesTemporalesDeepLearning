Pronósticos con Modelos Tradicionales
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ya conocemos que la serie de CO2 requiere tratamientos especiales debido
a que es una serie **No estacionaria**, por tanto de diferenciará en
primer orden

.. code:: ipython3

    #Obtener los datos
    import pandas as pd
    import statsmodels.api as sm
    
    data = sm.datasets.co2.load_pandas().data
    # Convertir el índice a datetime
    data.index = pd.to_datetime(data.index)
    data = data.asfreq("W-SAT")  
    
    #convertiremos la serie a frecuencia mensual
    y_m = data["co2"].resample("MS").mean().interpolate()  # mensual, inicio de mes
    y = y_m.copy()
    df = pd.DataFrame({"y": y})
    
    
    # diferenciar la serie para hacerla estacionaria
    y_diff = y.diff().dropna()
    df_diff = pd.DataFrame({"y_diff": y_diff})
    df_diff.head()



.. parsed-literal::

                         y
    1958-03-01  316.100000
    1958-04-01  317.200000
    1958-05-01  317.433333
    1958-06-01  316.529167
    1958-07-01  315.625000




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>y_diff</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1958-04-01</th>
          <td>1.100000</td>
        </tr>
        <tr>
          <th>1958-05-01</th>
          <td>0.233333</td>
        </tr>
        <tr>
          <th>1958-06-01</th>
          <td>-0.904167</td>
        </tr>
        <tr>
          <th>1958-07-01</th>
          <td>-0.904167</td>
        </tr>
        <tr>
          <th>1958-08-01</th>
          <td>-0.675000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    from plotly.subplots import make_subplots
    
    import plotly.graph_objects as go
    
    # Crear subplots con dos gráficos uno encima del otro
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Serie Original", "Serie en Diferencias"))
    
    # Agregar la serie original al primer gráfico
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name="Serie Original"), row=1, col=1)
    
    # Agregar la serie en diferencias al segundo gráfico
    fig.add_trace(go.Scatter(x=y_diff.index, y=y_diff, mode='lines', name="Serie en Diferencias"), row=2, col=1)
    
    # Configurar el diseño del gráfico
    fig.update_layout(
        title_text="Comparación de la Serie Original y la Serie en Diferencias",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        showlegend=False
    )
    
    fig.show()

.. image:: 2_modelos_tradionales_files/comparacion_diff.png
   


.. code:: ipython3

    # Split: últimos 60 meses como test
    H = 60
    train, test = y_diff.iloc[:-H], y_diff.iloc[-H:]

Aunque ya sabemos que se requiere de un modelo con estacionalidad,
ajustaremos otros para comparar, comencemos con un modelo AR de orden 1.

.. code:: ipython3

    from statsmodels.tsa.ar_model import AutoReg
    
    # Ajustar el modelo AR(1)
    model_ar1 = AutoReg(train, lags=1).fit()
    
    # Mostrar el resumen del modelo
    print(model_ar1.summary())


.. parsed-literal::

                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                    co2   No. Observations:                  465
    Model:                     AutoReg(1)   Log Likelihood                -573.624
    Method:               Conditional MLE   S.D. of innovations              0.833
    Date:                Mon, 13 Oct 2025   AIC                           1153.248
    Time:                        18:14:29   BIC                           1165.667
    Sample:                    05-01-1958   HQIC                          1158.136
                             - 12-01-1996                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.0293      0.039      0.755      0.450      -0.047       0.105
    co2.L1         0.7075      0.033     21.521      0.000       0.643       0.772
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            1.4134           +0.0000j            1.4134            0.0000
    -----------------------------------------------------------------------------


Análisis de Residuales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analizamos los residuales del modelo para garantizar que exista
**homocedasticidad**

.. code:: ipython3

    #Extraer los residuales
    residuals = model_ar1.resid
    # Graficar los residuales con plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=residuals.index, y=residuals, mode='markers', name='Residuales'))
    # Agregar una línea en y=0
    fig.add_trace(go.Scatter(x=residuals.index, y=[0]*len(residuals), mode='lines'))
    fig.update_layout(title='Residuales del Modelo AR(1)', xaxis_title='Fecha', yaxis_title='Residuales', showlegend=False)
    fig.show()

.. image:: 2_modelos_tradionales_files/resuduales_ar.png

La gráfica de residuales del modelo AR(1) muestra que los errores se
distribuyen aproximadamente alrededor de cero sin tendencia visible ni
patrones sistemáticos, lo que sugiere que el modelo ha capturado parte
importante de la dinámica temporal de la serie. Sin embargo, aún pueden
observarse algunas concentraciones o posibles autocorrelaciones leves.

.. code:: ipython3

    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Realizar la prueba de Ljung-Box sobre los residuales
    ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    # Mostrar los resultados
    print(ljung_box_result)


.. parsed-literal::

          lb_stat     lb_pvalue
    10  290.22345  1.808968e-56


Los resultados de la prueba de Ljung–Box (LB = 290.22, p ≈ 1.8 × 10⁻⁵⁶)
indican que los residuales del modelo AR presentan autocorrelación
significativa, ya que el p-valor es mucho menor que 0.05. Por tanto, se
rechaza la hipótesis nula de independencia y se concluye que los errores
no son ruido blanco, lo que sugiere que el modelo AR(1) no captura
completamente la dinámica temporal de la serie y requiere una
especificación más compleja (por ejemplo, aumentar el orden AR o
incorporar componentes MA o estacionales)

Isertemos el componente MA (ARMA (1,1))
==============================================

.. code:: ipython3

    from statsmodels.tsa.arima.model import ARIMA
    
    # Ajustar el modelo ARMA(1,1)
    model_arma11 = ARIMA(train, order=(1, 0, 1)).fit()
    
    # Mostrar el resumen del modelo
    print(model_arma11.summary())


.. parsed-literal::

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                    co2   No. Observations:                  465
    Model:                 ARIMA(1, 0, 1)   Log Likelihood                -546.117
    Date:                Tue, 14 Oct 2025   AIC                           1100.234
    Time:                        12:18:16   BIC                           1116.802
    Sample:                    04-01-1958   HQIC                          1106.755
                             - 12-01-1996                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.1095      0.121      0.902      0.367      -0.128       0.347
    ar.L1          0.5790      0.055     10.611      0.000       0.472       0.686
    ma.L1          0.3385      0.062      5.462      0.000       0.217       0.460
    sigma2         0.6121      0.047     12.896      0.000       0.519       0.705
    ===================================================================================
    Ljung-Box (L1) (Q):                   3.15   Jarque-Bera (JB):                 4.64
    Prob(Q):                              0.08   Prob(JB):                         0.10
    Heteroskedasticity (H):               1.11   Skew:                             0.12
    Prob(H) (two-sided):                  0.52   Kurtosis:                         2.57
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


Comportamiento de los Residuales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Los residuales de un modelo deben comportarse como **ruido blanco**, lo
que significa que no deben presentar autocorrelación ni patrones
sistemáticos. Matemáticamente, esto se expresa como:

.. math::


   E[\epsilon_t] = 0, \quad \text{Var}(\epsilon_t) = \sigma^2, \quad \text{Cov}(\epsilon_t, \epsilon_{t-k}) = 0 \; \forall k \neq 0

Donde :math:`\epsilon_t` representa los residuales en el tiempo
:math:`t`. Si los residuales no cumplen estas propiedades, el modelo
puede no ser adecuado para capturar la dinámica de la serie temporal.

.. code:: ipython3

    from scipy.stats import shapiro
    
    import plotly.figure_factory as ff
    
    # Graficar la distribución de los residuales
    fig = ff.create_distplot([residuals], group_labels=["Residuales"], show_hist=True, show_rug=False)
    fig.update_layout(title="Distribución de los Residuales", xaxis_title="Residuales", yaxis_title="Densidad")
    fig.show()
    
    # Realizar la prueba de Shapiro-Wilk
    stat, p_value = shapiro(residuals)
    print(f"Estadístico de Shapiro-Wilk: {stat}, p-valor: {p_value}")
    
    if p_value > 0.05:
        print("No se puede rechazar la hipótesis nula: los residuales parecen seguir una distribución normal.")
    else:
        print("Se rechaza la hipótesis nula: los residuales no siguen una distribución normal.")

.. image:: 2_modelos_tradionales_files/residuales_arma.png




.. parsed-literal::

    Estadístico de Shapiro-Wilk: 0.9895207146801304, p-valor: 0.0021362316746592067
    Se rechaza la hipótesis nula: los residuales no siguen una distribución normal.


El histograma de los residuales no muestra un comportamiento parecido a
yna campana gausiana

.. code:: ipython3

    # Extraer los residuales del modelo ARMA(1,1)
    residuals_arma = model_arma11.resid
    
    # Realizar la prueba de Ljung-Box sobre los residuales del modelo ARMA
    ljung_box_arma_result = acorr_ljungbox(residuals_arma, lags=[10], return_df=True)
    
    # Mostrar los resultados
    print(ljung_box_arma_result)


.. parsed-literal::

           lb_stat     lb_pvalue
    10  166.255102  1.652262e-30


Al igual que en el anterior, los residuales del modelo ARMA(1,1),
todavia presentan autocorrelaciones.


Componente Estacional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utilizamos el modelo SARIMA para capturar la estacionalidad en los datos.


Detalle Matemático del Modelo SARIMA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

El modelo SARIMA (Seasonal AutoRegressive Integrated Moving Average) es
una extensión del modelo ARIMA que incluye componentes estacionales. El
código:

.. code:: python

   model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

configura y ajusta un modelo SARIMA con los siguientes parámetros:

1. **``order=(1, 1, 1)``**:

   - **AR(1)**: Componente autorregresiva de orden 1, donde el valor
     actual depende linealmente de su valor pasado.

     .. math::


        X_t = \phi_1 X_{t-1} + \epsilon_t
   - **I(1)**: Componente de integración de orden 1, que diferencia la
     serie una vez para hacerla estacionaria.

     .. math::


        Y_t = X_t - X_{t-1}
   - **MA(1)**: Componente de media móvil de orden 1, que modela la
     relación entre el valor actual y los errores pasados.

     .. math::


        X_t = \theta_1 \epsilon_{t-1} + \epsilon_t

2. **``seasonal_order=(1, 1, 1, 12)``**:

   - **AR estacional (1)**: Componente autorregresiva estacional de
     orden 1, que considera la dependencia entre valores separados por
     un período estacional (12 meses en este caso).

     .. math::


        X_t = \Phi_1 X_{t-12} + \epsilon_t
   - **Diferenciación estacional (1)**: Diferenciación de la serie con
     un rezago estacional para eliminar tendencias estacionales.

     .. math::


        Y_t = X_t - X_{t-12}
   - **MA estacional (1)**: Componente de media móvil estacional de
     orden 1, que modela la relación entre el valor actual y los errores
     pasados separados por un período estacional.

     .. math::


        X_t = \Theta_1 \epsilon_{t-12} + \epsilon_t
   - **Período estacional (12)**: Indica que la estacionalidad tiene un
     ciclo de 12 meses.

En conjunto, el modelo SARIMA combina estas componentes para capturar
tanto las dinámicas temporales como las estacionales de la serie. Este
enfoque es útil para series temporales con patrones estacionales claros,
como las mediciones mensuales de CO₂ en Mauna Loa.

.. code:: ipython3

    # El modelo SARIMA realiza la diferenciación automáticamente para hacer la serie estacionaria por lo que usaremos la serie original
    
    H = 60
    train, test = df.iloc[:-H], df.iloc[-H:]


.. code:: ipython3

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Ajustar el modelo SARIMA
    sarima_model = SARIMAX(
       y,
        order=(1, 1, 1),             # parte no estacional
        seasonal_order=(1, 1, 1, 12),# parte estacional
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()
    
    # Mostrar el resumen del modelo
    print(sarima_model.summary())


.. parsed-literal::

                                         SARIMAX Results                                      
    ==========================================================================================
    Dep. Variable:                                co2   No. Observations:                  526
    Model:             SARIMAX(1, 1, 1)x(1, 1, 1, 12)   Log Likelihood                -102.151
    Date:                            Tue, 14 Oct 2025   AIC                            214.302
    Time:                                    12:59:38   BIC                            235.365
    Sample:                                03-01-1958   HQIC                           222.568
                                         - 12-01-2001                                         
    Covariance Type:                              opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.1575      0.122      1.296      0.195      -0.081       0.396
    ma.L1         -0.5004      0.109     -4.583      0.000      -0.714      -0.286
    ar.S.L12       0.0008      0.000      1.826      0.068   -6.23e-05       0.002
    ma.S.L12      -0.8605      0.027    -32.198      0.000      -0.913      -0.808
    sigma2         0.0858      0.006     15.485      0.000       0.075       0.097
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                 0.07
    Prob(Q):                              0.92   Prob(JB):                         0.97
    Heteroskedasticity (H):               0.86   Skew:                            -0.01
    Prob(H) (two-sided):                  0.32   Kurtosis:                         2.94
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).


Comprobemos residuales

.. code:: ipython3

    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import shapiro
    
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    
    # Extraer los residuales del modelo SARIMA, ignorando las primeras observaciones
    residuals_sarima = sarima_model.resid[15:]  # Ignorar las primeras 12 observaciones
    
    # Graficar los residuales
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=residuals_sarima.index, y=residuals_sarima, mode='markers', name='Residuales'))
    fig.add_trace(go.Scatter(x=residuals_sarima.index, y=[0]*len(residuals_sarima), mode='lines', name='Media Cero'))
    fig.update_layout(title='Residuales del Modelo SARIMA (Ajustados)', xaxis_title='Fecha', yaxis_title='Residuales', showlegend=False)
    fig.show()
    
    # Realizar la prueba de Ljung-Box para verificar no autocorrelación
    ljung_box_sarima_result = acorr_ljungbox(residuals_sarima, lags=[10], return_df=True)
    print("Resultados de la prueba de Ljung-Box (Ajustados):")
    print(ljung_box_sarima_result)
    
    # Graficar la distribución de los residuales
    fig = ff.create_distplot([residuals_sarima], group_labels=["Residuales"], show_hist=True, show_rug=False)
    fig.update_layout(title="Distribución de los Residuales del Modelo SARIMA (Ajustados)", xaxis_title="Residuales", yaxis_title="Densidad")
    fig.show()
    
    # Realizar la prueba de Shapiro-Wilk para verificar normalidad
    stat, p_value = shapiro(residuals_sarima)
    print(f"Estadístico de Shapiro-Wilk: {stat}, p-valor: {p_value}")
    
    if p_value > 0.05:
        print("No se puede rechazar la hipótesis nula: los residuales parecen seguir una distribución normal.")
    else:
        print("Se rechaza la hipótesis nula: los residuales no siguen una distribución normal.")




.. parsed-literal::

    Resultados de la prueba de Ljung-Box (Ajustados):
         lb_stat  lb_pvalue
    10  6.370597   0.783226


.. image:: 2_modelos_tradionales_files/residuales_sarima.png
   

.. parsed-literal::

    Estadístico de Shapiro-Wilk: 0.9935186700111294, p-valor: 0.027259616235327244
    Se rechaza la hipótesis nula: los residuales no siguen una distribución normal.


.. image:: 2_modelos_tradionales_files/hist_resis_sarima.png

Como regla práctica: si Ljung–Box y ARCH-LM pasan pero normalidad falla,
mantén la estructura SARIMA y corrige sólo la distribución (t-student o
bootstrapping) para bandas de predicción y tests.

Pronóstico con el modelo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Realizar el pronóstico para el conjunto de prueba
    forecast = sarima_model.get_forecast(steps=H)
    forecast_index = test.index
    forecast_values = forecast.predicted_mean
    
    # Graficar los valores reales y pronosticados
    fig = go.Figure()
    
    # Agregar los valores reales del conjunto de prueba
    fig.add_trace(go.Scatter(x=test.index, y=test['y'], mode='lines', name='Valores Reales'))
    
    # Agregar los valores pronosticados
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Pronóstico'))
    
    # Agregar las bandas de confianza
    conf_int = forecast.conf_int()
    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Intervalo de Confianza'
    ))
    
    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Pronóstico vs Valores Reales en test",
        xaxis_title="Fecha",
        yaxis_title="CO2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.show()

.. image:: 2_modelos_tradionales_files/pronostico_sarima.png


.. code:: ipython3

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    # Calcular métricas
    mae = mean_absolute_error(test['y'], forecast_values)
    mse = mean_squared_error(test['y'], forecast_values)
    rmse = np.sqrt(mse)
    
    
    # Mostrar resultados
    print(f"MAE (Error Absoluto Medio): {mae}")
    print(f"MSE (Error Cuadrático Medio): {mse}")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse}")


.. parsed-literal::

    MAE (Error Absoluto Medio): 7.979047387128087
    MSE (Error Cuadrático Medio): 63.99656988285457
    RMSE (Raíz del Error Cuadrático Medio): 7.999785614805847



.. code:: ipython3

    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    
    # Aplicar la prueba CUSUM a los residuales del modelo SARIMA
    cusum_stat, p_value, critical_values = breaks_cusumolsresid(residuals_sarima)
    
    # Mostrar los resultados de la prueba
    print(f"Estadístico CUSUM: {cusum_stat}")
    print(f"P-valor: {p_value}")
    print(f"Valores críticos: {critical_values}")
    
    # Graficar los resultados de la prueba CUSUM
    fig = go.Figure()
    
    # Agregar los residuales
    fig.add_trace(go.Scatter(x=residuals_sarima.index, y=residuals_sarima, mode='lines', name='Residuales'))
    
    # Agregar las líneas de los valores críticos
    fig.add_trace(go.Scatter(x=residuals_sarima.index, y=[critical_values[0]] * len(residuals_sarima), mode='lines', name='Límite Inferior', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=residuals_sarima.index, y=[critical_values[1]] * len(residuals_sarima), mode='lines', name='Límite Superior', line=dict(dash='dash', color='red')))
    
    # Agregar el sombreado entre los valores críticos
    fig.add_trace(go.Scatter(
        x=residuals_sarima.index.tolist() + residuals_sarima.index[::-1].tolist(),
        y=[critical_values[0]] * len(residuals_sarima) + [critical_values[1]] * len(residuals_sarima[::-1]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    
    # Configurar el diseño del gráfico con ajuste de escala
    fig.update_layout(
        title="Prueba CUSUM para Análisis de Estabilidad del Modelo",
        xaxis_title="Fecha",
        yaxis_title="Residuales",
        yaxis=dict(range=[min(residuals_sarima.min(), critical_values[0]) - 1, max(residuals_sarima.max(), critical_values[1]) + 1]),
        showlegend=True
    )
    
    fig.show()


.. parsed-literal::

    Estadístico CUSUM: 1.5037532831227998
    P-valor: 0.021722602433024974
    Valores críticos: [(1, 1.63), (5, 1.36), (10, 1.22)]




