import os
import tempfile
import urllib

import numpy as np
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def visualize_test():
    plotter = VisdomLinePlotter(env_name='tutorial')
    plotter.plot('loss', 'val', 'class loss', 1, 0.9)
    plotter.plot('loss', 'val', 'class loss', 2, 0.8)


def visualize_text(viz):

    global input
    assert viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'

    textwindow = viz.text('Hello World!')

    updatetextwindow = viz.text('Hello World! More text should be here')
    assert updatetextwindow is not None, 'Window was none'
    viz.text('And here it is', win=updatetextwindow, append=True)

    # text window with Callbacks
    txt = 'This is a write demo notepad. Type below. Delete clears text:<br>'
    callback_text_window = viz.text(txt)

    def type_callback(event):
        if event['event_type'] == 'KeyPress':
            curr_txt = event['pane_data']['content']
            if event['key'] == 'Enter':
                curr_txt += '<br>'
            elif event['key'] == 'Backspace':
                curr_txt = curr_txt[:-1]
            elif event['key'] == 'Delete':
                curr_txt = txt
            elif len(event['key']) == 1:
                curr_txt += event['key']
            viz.text(curr_txt, win=callback_text_window)

    viz.register_event_handler(type_callback, callback_text_window)


def visualize_matplotlib(viz):
    # matplotlib demo:
    try:
        import matplotlib.pyplot as plt
        plt.plot([1, 23, 2, 4])
        plt.ylabel('some numbers')
        viz.matplot(plt)
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)


def visualize_video(viz):
    # video demo:
    try:
        video = np.empty([256, 250, 250, 3], dtype=np.uint8)
        for n in range(256):
            video[n, :, :, :].fill(n)
        viz.video(tensor=video)
    except BaseException:
        print('Skipped video tensor example')

    try:
        # video demo:
        # download video from http://media.w3.org/2010/05/sintel/trailer.ogv
        video_url = 'http://media.w3.org/2010/05/sintel/trailer.ogv'
        videofile = os.path.join(tempfile.gettempdir(), 'trailer.ogv')
        urllib.request.urlretrieve(video_url, videofile)

        if os.path.isfile(videofile):
            viz.video(videofile=videofile, opts={'width': 864, 'height': 480})
    except BaseException:
        print('Skipped video file example')


def visualize_image(viz):
    # image demo
    viz.image(
        np.random.rand(3, 512, 256),
        opts=dict(title='Random!', caption='How random.'),
    )

    # image demo save as jpg
    viz.image(
        np.random.rand(3, 512, 256),
        opts=dict(title='Random image as jpg!', caption='How random as jpg.', jpgquality=50),
    )

    # image history demo
    viz.image(
        np.random.rand(3, 512, 256),
        win='image_history',
        opts=dict(caption='First random', store_history=True, title='Pick your random!'),
    )
    viz.image(
        np.random.rand(3, 512, 256),
        win='image_history',
        opts=dict(caption='Second random!', store_history=True),
    )


def visualize_images(viz):
    # grid of images
    viz.images(
        np.random.randn(20, 3, 64, 64),
        opts=dict(title='Random images', caption='How random.')
    )


def visualize_scatter(viz):
    #scatter plots
    Y = np.random.rand(100)
    old_scatter = viz.scatter(
        X=np.random.rand(100, 2),
        Y=(Y[Y > 0] + 1.5).astype(int),
        opts=dict(
            legend=['Didnt', 'Update'],
            xtickmin=-50,
            xtickmax=50,
            xtickstep=0.5,
            ytickmin=-50,
            ytickmax=50,
            ytickstep=0.5,
            markersymbol='cross-thin-open',
        ),
    )

    viz.update_window_opts(
        win=old_scatter,
        opts=dict(
            legend=['Apples', 'Pears'],
            xtickmin=0,
            xtickmax=1,
            xtickstep=0.5,
            ytickmin=0,
            ytickmax=1,
            ytickstep=0.5,
            markersymbol='cross-thin-open',
        ),
    )

    # 3d scatterplot with custom labels and ranges
    viz.scatter(
        X=np.random.rand(100, 3),
        Y=(Y + 1.5).astype(int),
        opts=dict(
            legend=['Men', 'Women'],
            markersize=5,
            xtickmin=0,
            xtickmax=2,
            xlabel='Arbitrary',
            xtickvals=[0, 0.75, 1.6, 2],
            ytickmin=0,
            ytickmax=2,
            ytickstep=0.5,
            ztickmin=0,
            ztickmax=1,
            ztickstep=0.5,
        )
    )

    # 2D scatterplot with custom intensities (red channel)
    viz.scatter(
        X=np.random.rand(255, 2),
        Y=(np.random.rand(255) + 1.5).astype(int),
        opts=dict(
            markersize=10,
            markercolor=np.random.randint(0, 255, (2, 3,)),
        ),
    )

    # 2D scatter plot with custom colors per label:
    viz.scatter(
        X=np.random.rand(255, 2),
        Y=(np.random.randn(255) > 0) + 1,
        opts=dict(
            markersize=10,
            markercolor=np.floor(np.random.random((2, 3)) * 255),
        ),
    )

    win = viz.scatter(
        X=np.random.rand(255, 2),
        opts=dict(
            markersize=10,
            markercolor=np.random.randint(0, 255, (255, 3,)),
        ),
    )

    # assert that the window exists
    assert viz.win_exists(win), 'Created window marked as not existing'

    # add new trace to scatter plot
    viz.scatter(
        X=np.random.rand(255),
        Y=np.random.rand(255),
        win=win,
        name='new_trace',
        update='new'
    )

    # 2D scatter plot with text labels:
    viz.scatter(
        X=np.random.rand(10, 2),
        opts=dict(
            textlabels=['Label %d' % (i + 1) for i in range(10)]
        )
    )
    viz.scatter(
        X=np.random.rand(10, 2),
        Y=[1] * 5 + [2] * 3 + [3] * 2,
        opts=dict(
            legend=['A', 'B', 'C'],
            textlabels=['Label %d' % (i + 1) for i in range(10)]
        )
    )


def visualize_bar(viz):
    # bar plots
    viz.bar(X=np.random.rand(20))
    viz.bar(
        X=np.abs(np.random.rand(5, 3)),
        opts=dict(
            stacked=True,
            legend=['Facebook', 'Google', 'Twitter'],
            rownames=['2012', '2013', '2014', '2015', '2016']
        )
    )
    viz.bar(
        X=np.random.rand(20, 3),
        opts=dict(
            stacked=False,
            legend=['The Netherlands', 'France', 'United States']
        )
    )


def visualize_hist(viz):
    # histogram
    viz.histogram(X=np.random.rand(10000), opts=dict(numbins=20))


def visualize_heatmap(viz):
    # heatmap
    viz.heatmap(
        X=np.outer(np.arange(1, 6), np.arange(1, 11)),
        opts=dict(
            columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
            colormap='Electric',
        )
    )


def visualize_surf(viz):
    # surface
    x = np.tile(np.arange(1, 101), (100, 1))
    y = x.transpose()
    X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
    viz.surf(X=X, opts=dict(colormap='Hot'))


def visualize_contour(viz):
    # contour
    x = np.tile(np.arange(1, 101), (100, 1))
    y = x.transpose()
    X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
    viz.contour(X=X, opts=dict(colormap='Viridis'))


def visualize_line(viz):
    # line plots
    viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

    Y = np.linspace(-5, 5, 100)
    viz.line(
        Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
        X=np.column_stack((Y, Y)),
        opts=dict(markers=False),
    )

    # line using WebGL
    webgl_num_points = 200000
    webgl_x = np.linspace(-1, 0, webgl_num_points)
    webgl_y = webgl_x ** 3
    viz.line(X=webgl_x, Y=webgl_y,
             opts=dict(title='{} points using WebGL'.format(webgl_num_points), webgl=True),
             win="WebGL demo")

    # line updates
    win = viz.line(
        X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
        Y=np.column_stack((np.linspace(5, 10, 10),
                           np.linspace(5, 10, 10) + 5)),
    )
    viz.line(
        X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
        Y=np.column_stack((np.linspace(5, 10, 10),
                           np.linspace(5, 10, 10) + 5)),
        win=win,
        update='append'
    )
    viz.line(
        X=np.arange(21, 30),
        Y=np.arange(1, 10),
        win=win,
        name='2',
        update='append'
    )
    viz.line(
        X=np.arange(1, 10),
        Y=np.arange(11, 20),
        win=win,
        name='delete this',
        update='append'
    )
    viz.line(
        X=np.arange(1, 10),
        Y=np.arange(11, 20),
        win=win,
        name='4',
        update='insert'
    )
    viz.line(X=None, Y=None, win=win, name='delete this', update='remove')

    viz.line(
        X=webgl_x + 1.,
        Y=(webgl_x + 1.) ** 3,
        win="WebGL demo",
        update='append',
        opts=dict(title='{} points using WebGL'.format(webgl_num_points * 2), webgl=True)
    )

    win = viz.line(
        X=np.column_stack((
            np.arange(0, 10),
            np.arange(0, 10),
            np.arange(0, 10),
        )),
        Y=np.column_stack((
            np.linspace(5, 10, 10),
            np.linspace(5, 10, 10) + 5,
            np.linspace(5, 10, 10) + 10,
        )),
        opts={
            'dash': np.array(['solid', 'dash', 'dashdot']),
            'linecolor': np.array([
                [0, 191, 255],
                [0, 191, 255],
                [255, 0, 0],
            ]),
            'title': 'Different line dash types'
        }
    )

    viz.line(
        X=np.arange(0, 10),
        Y=np.linspace(5, 10, 10) + 15,
        win=win,
        name='4',
        update='insert',
        opts={
            'linecolor': np.array([
                [255, 0, 0],
            ]),
            'dash': np.array(['dot']),
        }
    )

    Y = np.linspace(0, 4, 200)
    win = viz.line(
        Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
        X=np.column_stack((Y, Y)),
        opts=dict(
            fillarea=True,
            showlegend=False,
            width=800,
            height=800,
            xlabel='Time',
            ylabel='Volume',
            ytype='log',
            title='Stacked area plot',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        ),
    )

    # Assure that the stacked area plot isn't giant
    viz.update_window_opts(
        win=win,
        opts=dict(
            width=300,
            height=300,
        ),
    )


if __name__ == '__main__':
    port = 8097
    hostname = 'http://localhost'
    base_url = '/'

    viz = Visdom(port=port, server=hostname, base_url=base_url, username='', password='', use_incoming_socket=True)
    #visualize_matplotlib(viz)
    #visualize_video(viz)
    #visualize_image(viz)
    #visualize_images(viz)
    #visualize_scatter(viz)
    #visualize_bar(viz)
    #visualize_hist(viz)
    #visualize_heatmap(viz)
    #visualize_contour(viz)
    #visualize_surf(viz)
    visualize_line(viz)
