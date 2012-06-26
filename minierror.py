class Error(Exception):
    "Base exception class"

class InferTypeError(Error):
    "Raised when types of values cannot be inferred"

class UnmappableTypeError(Error):
    "Raised when a type cannot be mapped"

class UnpromotableTypeError(Error):
    "Raised when the compiler does not know how to promote two types."

class UnmappableFormatSpecifierError(Error):
    "Raised when a type cannot be mapped to a (printf) format specifier"

class InvalidTypeSpecification(Error):
    "Raised when a type is sliced incorrectly."