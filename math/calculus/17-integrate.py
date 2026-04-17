#!/usr/bin/env
def poly_integral(poly, c=0):
    """
    Calculates the integral of a polynomial.
    """
    # Validation: poly must be a list of numbers, c must be an int or float
    if not isinstance(poly, list) or not all(isinstance(n, (int, float)) for n in poly):
        return None
    if not isinstance(c, (int, float)):
        return None
    
    # If the input list is empty, the integral is just the constant [c]
    if len(poly) == 0:
        return [c]

    # Initialize the result list with the integration constant at index 0
    integral = [c]

    # Apply the power rule: integral of (a * x^i) is (a / (i+1)) * x^(i+1)
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        
        # Convert to integer if it's a whole number (e.g., 5.0 -> 5)
        if new_coeff == int(new_coeff):
            new_coeff = int(new_coeff)
            
        integral.append(new_coeff)

    # The prompt asks for the list to be as small as possible.
    # We remove trailing zeros from the end of the list, 
    # but keep at least the constant term.
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
