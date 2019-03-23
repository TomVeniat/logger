import numpy as np
import pprint
import logging

from collections import defaultdict

# optional visdom
try:
    import visdom
except ImportError:
    visdom = None

logger = logging.getLogger(__name__)

class Cache(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._x = []
        self._y = []

    def update(self, idx, value):
        self._x.append(idx)
        self._y.append(value)

    @property
    def x(self):
        return np.array(self._x)

    @property
    def y(self):
        return np.array(self._y)


class Plotter(object):

    def __init__(self, xp, visdom_opts, xlabel):
        super(Plotter, self).__init__()

        if visdom_opts is None:
            visdom_opts = {}

        assert visdom is not None, "visdom could not be imported"

        # visdom env is given by Experiment name unless specified
        if 'env' not in list(visdom_opts.keys()):
            visdom_opts['env'] = xp.name

        self.viz = visdom.Visdom(**visdom_opts)
        self.xlabel = None if xlabel is None else str(xlabel)
        self.windows = {}
        self.windows_opts = defaultdict(dict)
        self.append = {}
        self.cache = defaultdict(Cache)

    def set_win_opts(self, name, opts):
        self.windows_opts[name] = opts

    def _plot_xy(self, name, tag, x, y, time_idx=True):
        """
        Creates a window if it does not exist yet.
        Returns True if data has been sent successfully, False otherwise.
        """
        tag = None if tag == 'default' else tag

        if name in self.windows:
            # We just want to update the existing window (pane) with the new data points.
            # todo: Try catch this
            try:
                self.viz.line(Y=y, X=x, name=tag, opts=self.windows_opts[name], win=self.windows[name], update='append')
            except ConnectionError:
                return False
            return True

        opts = self.windows_opts[name]
        if 'xlabel' in opts:
            pass
        elif self.xlabel is not None:
            opts['xlabel'] = self.xlabel
        else:
            opts['xlabel'] = 'Time (s)' if time_idx else 'Index'

        # if 'legend' not in opts and tag:
        #     opts['legend'] = [tag]
        if 'title' not in opts:
            opts['title'] = name
        try:
            self.windows[name] = self.viz.line(Y=y, X=x, name=tag, opts=opts)
        except ConnectionError:
            return False
        return True


    def plot_xp(self, xp):

        config = xp.config.copy()
        if 'git_diff' in xp.config.keys():
            config.pop('git_diff')
        self.plot_config(config)

        for tag in sorted(xp.logged.keys()):
            for name in sorted(xp.logged[tag].keys()):
                self.plot_logged(xp.logged, tag, name)

    def plot_logged(self, logged, tag, name):
        xy = logged[tag][name]
        x = np.array(list(xy.keys())).astype(np.float)
        y = np.array(list(xy.values()))
        time_idx = not np.isclose(x, x.astype(np.int)).all()

        if not self._plot_xy(name, tag, x, y, time_idx):
            logger.warning('Failed to Plot {}_{}'.format(name, tag))

    def plot_metric(self, metric, send=True):
        name, tag = metric.name, metric.tag
        cache = self.cache[metric.name_id()]
        cache.update(metric.index.get(), metric.get())
        if send:
            sent = self._plot_xy(name, tag, cache.x, cache.y, metric.time_idx)
            
            # clear cache if data has been sent successfully
            if sent:
                cache.clear()
            else:
                logger.warning('#####')
                logger.warning('#####')
                logger.warning('#####')
                logger.warning('Problem when plotting tag:{}, name:{}'.format(tag, name))
                logger.warning('#####')
                logger.warning('#####')
                logger.warning('#####')

    def plot_config(self, config):
        msg = "<br />".join(["{}: {}".format(str(k), v)
                             for (k, v) in sorted(config.items(), key=lambda x: x[0])])
        # display dict on visdom
        self.viz.text(msg)
