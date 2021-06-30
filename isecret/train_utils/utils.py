# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import random
from time import time
import sys
import numpy as np
import cv2
import torch.nn.functional as F




'''
=============================
======= Train utils =========
=============================
'''
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=20):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            for image in images:
                image = torch.unsqueeze(image.data, 0)
                if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                    self.num_imgs = self.num_imgs + 1
                    self.images.append(image)
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

class TorchBar():
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty')
                                  and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
                current: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateful_metrics`,
                        `value_for_last_step` will be displayed as-is.
                        Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval
                    and self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)



def parallel_model(args, model):
    if args.overall.device == 'cpu':
        return model # TODO distributed training with cpu
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return  model # TODO: check the instance
    gpu = torch.cuda.current_device()
    model = model.cuda(device=gpu)
    if args.dist.n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    return model




'''
===========================
========== Log ============
===========================
'''

class Pbar(object):
    """ Progress bar with title and timer
    Arguments:
    name: the bars name.
    target: Total number of steps expected.
    width: Progress bar width on screen.
    Usage example
    ```
    import kpbar
    import time
    pbar = kpbar.Pbar('loading and processing dataset', 10)
    for i in range(10):
        time.sleep(0.1)
        pbar.update(i)
    ```
    ```output
    loading and processing dataset
    10/10  [==============================] - 1.0s
    ```
    """

    def __init__(self, name, target, width=30):
        self.name = name
        self.target = target
        self.start = time()
        self.numdigits = int(np.log10(self.target)) + 1
        self.width = width
        print(self.name)

    def update(self, step):

        bar = ('%' + str(self.numdigits) + 'd/%d ') % (step + 1, self.target)

        status = ""

        if step < 0:
            step = 0
            status = "negtive?...\r\n"

        stop = time()

        status = '- {:.1f}s'.format((stop - self.start))

        progress = float(step + 1) / self.target

        # prog
        prog_width = int(self.width * progress)
        prog = ''
        if prog_width > 0:
            prog += ('=' * (prog_width - 1))
            if step + 1 < self.target:
                prog += '>'
            else:
                prog += '='
        prog += ('.' * (self.width - prog_width))

        # text = "\r{0} {1} [{2}] {3:.0f}% {4}".format(self.name, bar, prog, pregress, status)

        text = "\r{0} [{1}] {2}".format(bar, prog, status)
        sys.stdout.write(text)
        if step + 1 == self.target:
            sys.stdout.write('\n')
        sys.stdout.flush()




def print_network(net, name, verbose=False):
    """Print the total number of parameters in the network and (if verbose) network architecture
    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
