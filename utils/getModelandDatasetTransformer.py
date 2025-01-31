from extension.PatchTST.PatchTST import DatasetForPatchTST
from extension.iTransformer.iTransformer import iTransformer
from extension.mTAN.mTAN import mTAN, DatasetForMTAN
from pypots.classification import BRITS, GRUD, Raindrop
from pypots.classification.brits.data import DatasetForBRITS
from pypots.classification.grud.data import DatasetForGRUD


def getModelandDatasetTransformer(model_type, data_info, batch_size=1024, EPOCH=None, saving_path=None):
    if model_type == "BRITS":
        model = BRITS(n_steps=data_info["ts_length"], n_features=data_info["ts_dimension"],
                      n_classes=data_info["num_class"], rnn_hidden_size=64, batch_size=batch_size,
                      epochs=EPOCH, model_saving_strategy="better", saving_path=saving_path)
        dataset_transformer = DatasetForBRITS
    elif model_type == "GRUD":
        model = GRUD(n_steps=data_info["ts_length"], n_features=data_info["ts_dimension"],
                     n_classes=data_info["num_class"], rnn_hidden_size=64, batch_size=batch_size,
                     epochs=EPOCH, model_saving_strategy="better", saving_path=saving_path)
        dataset_transformer = DatasetForGRUD
    elif model_type == "Raindrop":
        n_heads = 0
        for i in [4, 8]:
            if data_info["ts_dimension"] % i == 0: n_heads = i
        if n_heads == 0: n_heads = 2 - (data_info["ts_dimension"] % 2)
        model = Raindrop(n_steps=data_info["ts_length"], n_features=data_info["ts_dimension"],
                         n_classes=data_info["num_class"], batch_size=batch_size,
                         n_layers=3, d_model=data_info["ts_dimension"], d_inner=128, n_heads=n_heads,
                         dropout=0.3, epochs=EPOCH, model_saving_strategy="better", saving_path=saving_path)
        dataset_transformer = DatasetForGRUD
    elif model_type == "mTAN":
        model = mTAN(n_features=data_info["ts_dimension"], n_classes=data_info["num_class"],
                     batch_size=batch_size, epochs=EPOCH, saving_path=saving_path)
        dataset_transformer = DatasetForMTAN
    elif model_type == "iTransformer":
        model = iTransformer(n_steps=data_info["ts_length"], n_features=data_info["ts_dimension"],
                             n_classes=data_info["num_class"],  batch_size=batch_size,
                             n_layers=3, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=16, dropout=0.1,
                             attn_dropout=0.1, epochs=EPOCH, saving_path=saving_path)
        dataset_transformer = DatasetForPatchTST
    return model, dataset_transformer
