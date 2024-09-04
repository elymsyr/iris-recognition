from time import perf_counter
import sys
import io
from functools import wraps

class PrintLogger:
    """
    A custom logger that intercepts print statements, logs them to a file, 
    and also prints them to the console in real-time.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.original_stdout = sys.stdout  # Save the original stdout
    
    def write(self, message):
        # Write to the log file
        with open(self.file_path, 'a') as log_file:
            log_file.write(message)
        
        # Print to the console
        self.original_stdout.write(message)
    
    def flush(self):
        # This method is needed for compatibility with the sys.stdout
        self.original_stdout.flush()

def capture_prints_to_file(file_path='logger.txt'):
    """
    Decorator that captures all prints in a function and writes them to a specified file,
    while also printing them to the console dynamically.
    
    :param file_path: The path to the log file. Defaults to 'logger.txt'.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Replace stdout with custom logger
            logger = PrintLogger(file_path)
            original_stdout = sys.stdout
            sys.stdout = logger

            try:
                # Call the original function with all its arguments
                result = func(*args, **kwargs)
                return result

            finally:
                # Restore the original stdout
                sys.stdout = original_stdout

        return wrapper
    return decorator

def counter(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} ended in {end - start:.6f} seconds.")
        return result
    return wrapper

class DummyFile(io.StringIO):
    def write(self, *args, **kwargs):
        pass  # Override write to suppress output

def suppress_print(func):
    def wrapper(*args, **kwargs):
        # Backup the real stdout
        real_stdout = sys.stdout
        try:
            # Redirect stdout to the dummy file
            sys.stdout = DummyFile()
            # Run the function
            return func(*args, **kwargs)
        finally:
            # Restore the real stdout
            sys.stdout = real_stdout
    return wrapper