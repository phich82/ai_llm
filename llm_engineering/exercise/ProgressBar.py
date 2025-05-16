import threading
import time
import sys, time, random


class SafeThread(threading.Thread):
    def __init__(self, target=None, args: list=[], kwargs: dict=None):
        print(args)
        super(SafeThread, self).__init__(target=target, args=args, kwargs=kwargs)
        self.exception = None

    def run(self) -> None:
        try:
            super(SafeThread, self).run()
        except Exception as e:
            self.exception = e
            print(e)

    def join(self, *args, **kwargs) -> None:
        super(SafeThread, self).join(*args, **kwargs)
        if self.exception:
            raise self.exception


class CustomThread(threading.Thread):

    def __init__(self, target=None, args: list=[], kwargs: dict | None=None):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.exception = None

    def run(self):
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        except Exception as exception:
            thread = threading.current_thread()
            self.exception = exception
            print(exception)
        finally:
            del self._target, self._args, self._kwargs


class ProgressBar:

    __is_done = False

    def __init__(self):
        self.__is_done = False

    def start(self, target=None, args: list=[], kwargs: dict=None):
        thread = SafeThread(target=target, args=args, kwargs=kwargs)
        # thread = CustomThread(target=target, args=args, kwargs=kwargs)
        thread.start()

        # Show the waiting content
        while not self.__is_done:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(0.1)  # Check every 100ms

            # Exit when error occured
            if thread.exception != None:
                self.done()

        sys.stdout.write('\n')
        sys.stdout.flush()

    def done(self):
        self.__is_done = True