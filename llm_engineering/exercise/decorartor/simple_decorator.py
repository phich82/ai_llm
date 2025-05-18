# Define a simple decorator function (as warapper function)
def simple_decorator(func, a:str=None):
    def wrapper_func():
        print("Hello, this is before function execution")
        print(a)
        func()
        print("This is after function execution")
    return wrapper_func

@simple_decorator
def display():
    print("This is inside the function !!")

# Calling the decorated function
display()


def apply_lambda(func, value):
    return func(value)