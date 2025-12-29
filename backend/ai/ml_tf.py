import os, numpy as np, tensorflow as tf

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pattern_tf.h5')

def _build_model(T=256, F=10, classes=12):
    inp = tf.keras.Input(shape=(T,F))
    x = tf.keras.layers.Conv1D(64,5,padding='same',activation='relu')(inp)
    x = tf.keras.layers.Conv1D(64,5,padding='same',activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)

_model = _build_model()
_loaded = False
_classes = [
    'DoubleTop','DoubleBottom','HeadShoulders','InvHeadShoulders',
    'TriangleAsc','TriangleDesc','Flag','Pennant','WedgeRise','WedgeFall',
    'SupportRes','None'
]

def _ensure_loaded():
    global _loaded, _model
    if _loaded: return
    if os.path.exists(WEIGHTS_PATH):
        _model.load_weights(WEIGHTS_PATH)
    _loaded = True

def infer_patterns_tf(x: np.ndarray):
    _ensure_loaded()
    if x.ndim != 2: raise ValueError('expected [T,F]')
    x = x[np.newaxis, ...]
    probs = _model.predict(x, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    return {
        'backend': 'tf',
        'class': _classes[best_idx],
        'probs': { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
    }
