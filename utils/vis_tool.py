import time
import numpy as np
import matplotlib
import torch as t
import visdom

matplotlib.use('Agg')
from matplotlib import pyplot as plot


def vis_image(img, ax=None):
    """Visualize a color image.
        Args:
            img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
                This is in RGB format and the range of its value is
                :math:`[0, 255]`.
            ax (matplotlib.axes.Axis): The visualization is displayed on this
                axis. If this is :obj:`None` (default), a new axis is created.
        Returns:
            ~matploblib.axes.Axes:
            Returns the Axes object with the plot for further tweaking.
        """

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax

def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it
    @param fig: a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)

def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plot.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.

def vis_bbox(img, bbox, text=None, color=None, ax=None):
    """Visualize bounding boxes inside image.
    :param img: (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
    :param bbox: (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(x_{min}, y_{min}, x_{max}, y_{max})` in the second axis.
    :param text: (~list of string): A list of text for each bbox.
    :param color: A list of color for each bbox.
    :param ax: (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    :returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """
    ax = vis_image(img, ax)
    if len(bbox) == 0:
        return ax
    if color is None:
        color = ['red' for _ in range(len(bbox))]
    for i, bb in enumerate(bbox):
        color_i = color[i]
        xy = (bb[0], bb[1])
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(xy, width, height, fill=False, edgecolor=color_i, linewidth=2))

        if text is not None:
            ax.text(bb[0], bb[1], text[i], style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data

class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by self.line, self.scatter, self._sent, etc.
    due to the implementation of `__getattr__`
    """
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs

        # e.g.('loss',23) the 23th value of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, index=0, showlegend=False, **kwargs):
        """
        self.plot('loss',1.00)
        index:          index of lines plotted in this figure
        """
        x = self.index.get(name, 0)-index
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name, showlegend=showlegend),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        !!don't ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~!!
        """
        self.vis.images(t.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self

    def img_heatmap(self, name, X):
        self.vis.heatmap(X, win=name)

    def image_bbox(self, name, image, bbox, text, color, ax=None):
        img_data = visdom_bbox(image, bbox, text, color)
        self.img(name, img_data)


if __name__=='__main__':
    vis = Visualizer()
    vis.img_heatmap('heatmap', (np.random.rand(180, 230)-0.5)*300)
