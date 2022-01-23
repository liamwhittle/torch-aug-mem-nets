"""
Visualisations for various analysis tasks in understanding the NTM
"""
from torch import Tensor
from PIL import Image, ImageDraw
import io
from IPython.display import Image as IPythonImage


def im_to_png_bytes(im):
    png = io.BytesIO()
    im.save(png, 'PNG')
    return bytes(png.getbuffer())


def create_grid(array: Tensor) -> Image:
    """
    Displays 2-D float vector as a grid heatmap. Very useful for plotting memory contents, sequential read and writes,
    controller outputs, network outputs, and more!
    :param array: Tensor(heigh, width)
    :return: None
    """

    image_height = 250
    image_width = 750

    im = Image.new("RGB", (image_width, image_height), (128, 128, 128))
    draw = ImageDraw.Draw(im)

    square_dim = int(image_height / array.shape[1])

    # we are sort of putting the tensor on its side
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            x0 = x * square_dim
            y0 = y * square_dim
            x1 = (x + 1) * square_dim
            y1 = (y + 1) * square_dim
            color = (int(float(array[x][y]) * 255), int(float(array[x][y]) * 255), int(float(array[x][y]) * 255))
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=color)
    return im


def show_grid_jupyter(t: Tensor) -> IPythonImage:
    return IPythonImage(im_to_png_bytes(create_grid(t)))
