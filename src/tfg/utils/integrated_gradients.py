from pathlib import Path
from typing import Optional, Union
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


"""
Most of the code in this file has been taken or adapted from this page:
https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
"""

def interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    
    return images


def compute_gradients(model, images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    
    return integrated_gradients


@tf.function
def integrated_gradients(model, image, target_class_idx,
                         baseline = None, input_layer_idx: Optional[int] = 0,
                         m_steps=50, batch_size=32):
    if baseline is None:
        image_size = model.layers[input_layer_idx].input_shape[0][1:]
        baseline = tf.zeros(shape=image_size)
    
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                        image=image,
                                                        alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(model=model, images=interpolated_path_input_batch,
                                        target_class_idx=target_class_idx)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def plot_img_attributions(model,
                          img_arr,
                          target_class_idx,
                          true_label, pred_label,
                          baseline = None,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4,
                          plot_baseline: Optional[bool] = False,
                          show: Optional[bool] = True):
    image = tf.convert_to_tensor(img_arr)

    attributions = integrated_gradients(
        model=model,
        baseline=baseline,
        image=image,
        target_class_idx=target_class_idx,
        m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    if plot_baseline:
        nrows, ncols = 2, 2
        fig_size = (14, 10)
    else:
        nrows, ncols = 1, 3
        fig_size = (21, 5)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=fig_size)

    if plot_baseline:
        axs[0, 0].set_title('Baseline image')
        axs[0, 0].imshow(baseline)
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Original image')
        axs[0, 1].imshow(image)
        axs[0, 1].axis('off')

        axs[1, 0].set_title('Attribution mask')
        axs[1, 0].imshow(attribution_mask, cmap=cmap)
        axs[1, 0].axis('off')

        axs[1, 1].set_title('Overlay')
        axs[1, 1].imshow(attribution_mask, cmap=cmap)
        axs[1, 1].imshow(image, alpha=overlay_alpha)
        axs[1, 1].axis('off')
    else:
        axs[0, 0].set_title('Original image')
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Attribution mask')
        axs[0, 1].imshow(attribution_mask, cmap=cmap)
        axs[0, 1].axis('off')

        axs[0, 2].set_title('Overlay')
        axs[0, 2].imshow(attribution_mask, cmap=cmap)
        axs[0, 2].imshow(image, alpha=overlay_alpha)
        axs[0, 2].axis('off')

    fig.suptitle(f"Integrated gradients - PRED = {pred_label} - TRUE = {true_label}", fontsize=26)
    
    if show:
        plt.show()
    
    return fig, axs




def plot_saliency_and_img_attributions(model, img_arr, target_class_idx, true_label, pred_label,
                          baseline = None, m_steps=50, cmap="inferno", overlay_alpha=0.4,
                          title_suffix: Optional[str] = None, show: Optional[bool] = True,
                          save_to_file: Optional[Union[str, Path]] = None):
    # Saliency part
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_parts = [
        f"TRUE = {true_label}",
        f"PRED = {pred_label}",
    ]
    if title_suffix is not None:
        title_parts.append(title_suffix)
    
    title = " / ".join(title_parts)
    fig.suptitle(title, fontsize=26)

    # Plot original image
    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("Original image")
    # Plot saliency
    #i = axes[0, 1].imshow(grad_eval, cmap=cmap, alpha=0.8)
    i = axes[0, 1].imshow(grad_eval, cmap=cmap)
    #axes[1, 1].imshow(attribution_mask, cmap=cmap)
    axes[0, 1].imshow(img_arr, alpha=overlay_alpha)

    axes[0, 1].set_title("Saliency map")
    fig.colorbar(i, ax=axes[0, 1])
    
    # Integrated gradients part
    image = tf.convert_to_tensor(img_arr)

    attributions = integrated_gradients(
        model=model,
        baseline=baseline,
        image=image,
        target_class_idx=target_class_idx,
        m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    axes[1, 0].set_title('Attribution mask')
    axes[1, 0].imshow(attribution_mask, cmap=cmap)
    axes[1, 0].axis('off')

    axes[1, 1].set_title('Integrated Gradients (Overlay original with attribution)')
    axes[1, 1].imshow(attribution_mask, cmap=cmap)
    axes[1, 1].imshow(image, alpha=overlay_alpha)
    axes[1, 1].axis('off')

    if save_to_file is not None:
        plt.savefig(str(save_to_file))

    if show:
        plt.show()
    
    return fig, axes

