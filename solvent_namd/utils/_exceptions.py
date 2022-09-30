"""
STATUS: NOT TESTED

"""


class InvalidInputError(Exception):
    """
    Thrown when an invalid input is given in the input file.

    """
    def __init__(self, message: str) -> None:
        super().__init__(message)
