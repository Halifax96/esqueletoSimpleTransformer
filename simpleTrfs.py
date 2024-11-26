!pip3 install spacy simpletransformers pandas

from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
from spacy.tokens import DocBin
import pickle


model_args = NERArgs()
model_args.labels_list = ["-","O", "I-LEETSPEAK", "B-LEETSPEAK", "B-INV_CAMO", "I-INV_CAMO", "B-PUNCT_CAMO", "I-PUNCT_CAMO","B-MIX", "I-MIX"]
model_args.overwrite_output_dir = True
model_args.output_dir = "/content/salida/"

model = NERModel(
    "roberta",
    "roberta-base",
    args=model_args,
    use_cuda=True,
)

def inspect_pickle_file(pickle_file):
    """Inspecciona el contenido del archivo .pkl."""
    print(f"Inspeccionando el archivo {pickle_file}...")
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    print("\nTipo de datos:", type(data))  # Tipo de datos cargados
    if isinstance(data, dict):
        print("\nLlaves del diccionario:", data.keys())
    elif isinstance(data, list):
        print("\nPrimeros elementos de la lista:", data[:5])
    else:
        print("\nContenido desconocido. Ejemplo:", str(data)[:500])  # Muestra los primeros 500 caracteres

    return data

# Inspecciona el archivo
pickle_file = "/content/NER_TRAIN_DATA_iob_format.pkl"
data = inspect_pickle_file(pickle_file)


def export_to_csv(data, output_csv):
    """Convierte los datos cargados a un DataFrame y los exporta como CSV."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(output_csv, index=False)
        print(f"Archivo guardado como {output_csv}")
    elif isinstance(data, list) and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Archivo guardado como {output_csv}")
    else:
        print("No se pudo convertir a CSV. Los datos no son compatibles.")

# Exporta a CSV
output_csv = "/content/NER_TRAIN_DATA_iob_format.csv"
export_to_csv(data, output_csv)




train_df = pd.DataFrame(data)
train_df.columns = ["sentence_id", "words", "labels"]


model.train_model(train_df)

result, model_outputs, wrong_preds = model.eval_model(train_df)
print(result)