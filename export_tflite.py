"""Optional real full-int8 TinyML model export. Requires: pip install tensorflow>=2.16

This trains a small Keras reconstruction model and exports a quantized TFLite model.
A model alone is not firmware: microcontroller integration and on-device benchmarks remain necessary.
"""
from pathlib import Path
import json
import numpy as np
from tinyguard.core import simulate, split_normal_train_test

def main():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit('Install optional TensorFlow: pip install tensorflow>=2.16') from exc
    tf.keras.utils.set_random_seed(42)
    train, test, y = split_normal_train_test(simulate(1200))
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1
    z = ((train-mean)/std).astype('float32')
    network = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4),
    ])
    network.compile(optimizer='adam', loss='mse')
    network.fit(z,z,epochs=45,batch_size=32,verbose=0,validation_split=.15)
    reconstruction = network.predict(z,verbose=0)
    threshold = float(np.quantile(np.mean((z-reconstruction)**2,axis=1),.97))
    converter = tf.lite.TFLiteConverter.from_keras_model(network)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative():
        for item in z[:200]:
            yield [item.reshape(1,4)]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    out = Path('artifacts'); out.mkdir(exist_ok=True)
    model_bytes = converter.convert()
    (out/'tinyguard_int8.tflite').write_bytes(model_bytes)
    (out/'preprocessing.json').write_text(json.dumps({'features':['temperature','vibration','humidity','current'], 'mean':mean.tolist(), 'std':std.tolist(), 'reconstruction_mse_threshold':threshold, 'quantized_input_requires_scale_zero_point':True},indent=2))
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    inp, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert inp['dtype'] == np.int8 and output['dtype'] == np.int8
    print('Exported:',out/'tinyguard_int8.tflite','bytes:',len(model_bytes))
    print('Int8 input:',inp['quantization'],'output:',output['quantization'])
    print('NOTE: training threshold is based on float Keras outputs; recalibrate on quantized inference before deployment.')

if __name__ == '__main__': main()
