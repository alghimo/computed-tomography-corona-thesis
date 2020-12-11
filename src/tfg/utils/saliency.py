from pathlib import Path
from typing import Optional, Union
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_saliency_from_img_arr(model: keras.Model, img_arr: np.ndarray,
                               true_label: str, pred_label: str,
                               save_to_file: Optional[Union[str, Path]] = None,
                               show: Optional[bool] = True):
    img = img_arr.reshape((1, *img_arr.shape))
    y_pred = model.predict(img)
    """
    The highest class score is at index 0, which is equivalent to the CP class.
    We can calculate the gradient with respect to the top class score to see
    which pixels in the image contribute the most:
    """
    images = tf.Variable(img, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max = np.max(dgrad_abs, axis=3)[0]
    ## normalize to range between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max), np.max(dgrad_max)
    grad_eval = (dgrad_max - arr_min) / (arr_max - arr_min + 1e-18)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img_arr)
    i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    fig.suptitle(f"Saliency plot - PRED = {pred_label} - TRUE = {true_label}", fontsize=26)

    if save_to_file is not None:
        plt.savefig(str(save_to_file))

    if show:
        plt.show()
    
    return fig, axes


def plot_saliency(model: keras.Model, img_file: Union[str, Path],
                  true_label: str, pred_label: str,
                  input_layer_idx: Optional[int] = 0,
                  save_to_file: Optional[Union[str, Path]] = None,
                  show: Optional[bool] = True):
    image_size = model.layers[input_layer_idx].input_shape[0][1:]
    loaded_img = keras.preprocessing.image.load_img(img_file, target_size=img_size)
    img_arr = keras.preprocessing.image.img_to_array(loaded_img)
    
    return plot_saliency_from_img_arr(model=model, img_arr=img_arr,
                               true_label=true_label, pred_label=pred_label,
                               save_to_file=save_to_file,
                               show=show)
