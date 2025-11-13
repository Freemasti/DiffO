import torch
from compressai.zoo import cheng2020_anchor

def recon(model_name, quality, img_tensor):
    """
    Perform image compression and reconstruction using the specified LIC model and quality.

    Args:
        model_name (str): Name of the model in compressai (e.g., 'bmshj2018-hyperprior').
        quality (int): Quality level (usually between 1 and 8 for most compressai models).
        img (Tensor): Input image tensor of shape (b, c, h, w), values in [0, 1] range.

    Returns:
        Tensor: Reconstructed image after compression.
    """
    # Load the specified pre-trained model with the given quality level
    model = cheng2020_anchor(quality=quality, pretrained=True).eval().cuda()

    with torch.no_grad():
        # Compress and decompress the image using the loaded model
        reconstruction = model(img_tensor)

    return reconstruction["x_hat"]
