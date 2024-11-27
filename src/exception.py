import sys
import traceback

def error_message_detail(error, error_detail: sys):
    try:
        # Extract traceback details from the sys.exc_info() tuple
        exc_type, exc_value, exc_tb = error_detail.exc_info()

        # Check if the traceback is available
        if exc_tb is None:
            # If no traceback is available, simply return the error message
            return f"Error occurred: {str(error)}"

        # Extract filename, line number, and error message from the traceback
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = str(error)

        # Format and return the detailed error message
        return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{error_message}]"
    except Exception as e:
        # In case something goes wrong, return a generic error message
        return f"Error occurred while processing the error message: {str(e)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Call the base class constructor with the error message
        super().__init__(error_message)
        
        # Store the detailed error message generated from error_message_detail
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
