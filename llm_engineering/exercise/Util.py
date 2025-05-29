

class Util:

    @staticmethod
    def empty(value: any):
        if value is None:
            return True
        if isinstance(value, str):
            return value == ''
        if isinstance(value, list) or isinstance(value, dict):
            return len(value) < 1
        raise Exception('Data type not support.')

    @staticmethod
    def not_empty(value: any):
        return not Util.empty(value)

    @staticmethod
    def is_fn(fn):
        return callable(fn)
        # return hasattr(fn, '__call__')
        # return isinstance(fn, types.FunctionType)