import pandas as pd
import joblib

def load_objects():
    model = joblib.load('model.pkl')
    imputer = joblib.load('impute.pkl')
    encode = joblib.load('encode.pkl')
    scale = joblib.load('scale.pkl')
    return model, imputer, encode, scale


def preprocess(data,impute, encode, scale):
    new_ = pd.DataFrame(data).T
    new_.columns = ['name','fuel_type', 'km_driven']
    new_['km_driven'] = impute.transform(new_[['km_driven']])
    res_encode_cat = encode.transform(new_[['name','fuel_type']])
    new_['km_driven'] = scale.transform(new_[['km_driven']])
    cat_out = pd.DataFrame(res_encode_cat, columns=encode.get_feature_names_out())
    final_preprocessed = pd.concat([cat_out,new_['km_driven']],axis =1)
    return final_preprocessed