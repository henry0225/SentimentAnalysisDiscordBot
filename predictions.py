import numpy as np
def predict(model, text):
    return model.predict([np.array(text)])