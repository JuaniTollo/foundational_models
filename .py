from datasets import load_from_disk

import pdb
pdb.set_trace()
# Cargar el dataset desde la carpeta donde está almacenado
dataset = load_from_disk("datasets/newExchangeRate")

# Mostrar las columnas disponibles
#print("Columnas:", dataset.column_names)

# Mostrar los primeros 5 ejemplos
#print("Primeros 5 ejemplos:", dataset[:5])
