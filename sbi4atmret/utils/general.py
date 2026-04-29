



def transform_uniform(x, a, b, c, d):
        # Check if x is within the original range
        if a <= x <= b:
            # Apply the transformation formula
            y = c + ((x - a) * (d - c)) / (b - a)
            return y
        else:
            raise ValueError(f"x must be in the range [{a}, {b}]")

