import types


class Util:

    @staticmethod
    def is_fn(fn):
        return callable(fn)
        # return hasattr(fn, '__call__')
        # return isinstance(fn, types.FunctionType)