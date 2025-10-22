Instalación y Configuración
===========================

Requisitos del Sistema
----------------------

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.8+** (recomendado 3.9 o superior)
- **pip** (gestor de paquetes de Python)
- **Git** (para clonar el repositorio)

Instalación del Entorno
-----------------------

1. Crea un ambiente virtual
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Crear ambiente virtual
   python -m venv venv
   
   # Activar ambiente virtual
   # En macOS/Linux:
   source venv/bin/activate
   
   # En Windows:
   venv\\Scripts\\activate

2. Instalar las dependencias del proyecto:

   Ejecuta el siguiente comando para instalar las dependencias listadas en el archivo `requirements.txt`:

   .. code-block:: bash

      pip install -r requirements.txt

   Puedes consultar el archivo `requirements.txt` directamente aquí: `requirements.txt <requirements.txt>`_.

Dependencias Principales
------------------------

El proyecto utiliza las siguientes librerías:

Core Libraries
~~~~~~~~~~~~~~

.. code-block:: text

   numpy>=1.21.0
   pandas>=1.3.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   scikit-learn>=1.0.0
   plotly>=5.0.0
   statsmodels>=0.13.0

Deep Learning
~~~~~~~~~~~~~

.. code-block:: text

   tensorflow>=2.8.0
   keras>=2.8.0

Series Temporales
~~~~~~~~~~~~~~~~~

.. code-block:: text

   statsmodels>=0.13.0
   sktime>=0.11.0
   tsfresh>=0.19.0

Visualización
~~~~~~~~~~~~~

.. code-block:: text

   plotly>=5.0.0
   bokeh>=2.4.0

Jupyter
~~~~~~~

.. code-block:: text

   jupyter>=1.0.0
   ipywidgets>=7.6.0


Configuración de Jupyter
------------------------

Para usar los notebooks de manera óptima:

.. code-block:: bash

   # Instalar kernel de Jupyter
   python -m ipykernel install --user --name=timeseries --display-name="Series Temporales"
   
   # Instalar extensiones útiles
   pip install jupyter_contrib_nbextensions
   jupyter contrib nbextension install --user

Problemas Comunes
----------------

**Error de dependencias**
  Si encuentras conflictos de versiones, usa un ambiente virtual limpio

**GPU no detectada**
  Verifica la instalación de drivers CUDA para TensorFlow/PyTorch

**Jupyter no encuentra el kernel**
  Asegúrate de haber activado el ambiente virtual antes de instalar el kernel

.. tip::
   Si trabajas en Google Colab, muchas de estas dependencias ya están preinstaladas.
   Solo necesitarás instalar algunos paquetes específicos.