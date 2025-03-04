# Text-to-SQL_TFM
Trabajo de fin de master IA en UNIR

Tal como se ha especificado en el documento de la memoria, a continuación se explican algunos detalles para poder ejecutar y probar la aplicación desarrollada. Antes de todo, no se olvide instalar todas las librerias presentes en el fichero "requirements.txt".

## Instalación del modelo
Una vez descargado el código, dirijense a la carpeta model_assets. Allí encuentra un ficheo llamado "readme.txt" donde está el enlace para descargar el modelo Text-to-SQL desarrollado. En esa misma carpeta hay que poner tanto el modelo como sus dos ficheros con tokenizadores en formato json. Esto se ha hecho, debido a que el modelo es demasiado grande (en torno a 400 MB) y github no lo admite.

## Datos para entrenamiento y ejecución
En la carpeta "dataset" encontrará el fichero CSV usado para el entrenamiento del modelo y también un script de sql que, previamente a la ejecución del programa, lo tendrá que ejecutar en una base de datos postgreSQL, en local, ya que la herramienta busca la base de datos en local. Para modificar la conexión, modifique el fichero DB_Connection.py que está en la carpeta scripts. La variable DB_CONN contiene los datos de la conexión con la base de datos en local.

## Cuaderno con el código para la red neuronal
En la carpeta dataset, encontrará el cuaderno con el código hecho para el desarrollo del modelo.

## Acceso al modelo de traducción
Para poder probar la posibilidad de hacer una consulta en castellano, hace falta tener un token de acceso en HuggingFace. Una vez obtenido el token de acceso, se tiene que poner en el fichero .env sustituyendo la palabra <i>your_access_token</i>. Como se imagina, este token es secreto y bajo ninguna circunstancia se debe revelar.