import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Infer single image from TF checkpoint')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', default='hyperspectral-0.001-2conv-basic.model', help='Model base name')
    parser.add_argument('--img-size', type=int, default=50, help='Image size (square)')
    args = parser.parse_args()

    tf.compat.v1.disable_v2_behavior()
    IMG_SIZE = args.img_size
    model_base = args.model
    meta_path = model_base + '.meta'

    if not os.path.exists(meta_path):
        print(f'Meta graph not found: {meta_path}')
        return

    tf.compat.v1.reset_default_graph()
    saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_base)
        graph = tf.compat.v1.get_default_graph()

        # Find input tensor: try common name 'input:0' then first placeholder
        try:
            input_tensor = graph.get_tensor_by_name('input:0')
        except Exception:
            placeholders = [op.outputs[0] for op in graph.get_operations() if op.type == 'Placeholder']
            if not placeholders:
                print('No input placeholder found in graph.')
                return
            input_tensor = placeholders[0]

        # Find output tensor: look for softmax op or common names
        out_tensor = None
        for op in graph.get_operations():
            if 'softmax' in op.name.lower() or op.type.lower() == 'softmax':
                out_tensor = op.outputs[0]
                break

        if out_tensor is None:
            for name in ('targets:0', 'output:0', 'pred:0', 'prediction:0'):
                try:
                    out_tensor = graph.get_tensor_by_name(name)
                    break
                except Exception:
                    pass

        if out_tensor is None:
            print('Could not auto-detect output tensor. Available operations:')
            for op in graph.get_operations()[-40:]:
                print('  ', op.name, op.type)
            return

        # Load and preprocess image
        img = cv2.imread(args.image)
        if img is None:
            print('Failed to read image:', args.image)
            return
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype(np.float32)

        feed = {input_tensor: [img]}
        pred = sess.run(out_tensor, feed_dict=feed)
        pred = np.array(pred)
        print('Raw output:', pred)
        print('Predicted class:', int(np.argmax(pred, axis=1)[0]))

if __name__ == '__main__':
    main()
