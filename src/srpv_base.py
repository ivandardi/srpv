import abc


class srpv(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError('users must define __init__ to use this base class')

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError('users must define run to use this base class')

    @abc.abstractmethod
    def crop(self):
        raise NotImplementedError('users must define crop to use this base class')
