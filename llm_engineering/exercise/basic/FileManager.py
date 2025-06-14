from contextlib import contextmanager


# Using the custom context manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        """Acquires the resource and returns it (resource allocation)

        Returns:
            _type_: _description_
        """
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        """Releases the resource when the block exits (resource deallocation)

        Args:
            exc_type (_type_): _description_
            exc_value (_type_): _description_
            traceback (_type_): _description_
        """
        self.file.close()


# Using the contextlib Module
@contextmanager
def open_file(filename, mode):
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()

# Using the generator-based context manager
with open_file('example.txt', 'w') as file:
    file.write('Hello, World!')

if __name__ == '__main__':
    with FileManager('example.txt', 'w') as file:
        file.write('Hello, World!')