



def transform_uniform(x, a, b, c, d):
        # Check if x is within the original range
        if a <= x <= b:
            # Apply the transformation formula
            y = c + ((x - a) * (d - c)) / (b - a)
            return y
        else:
            raise ValueError(f"x must be in the range [{a}, {b}]")




def instrument_from_simname(sim_name: str) -> str:
    """
    Extract instrument name from a sim_name key.

    Examples:
        "cloudfree_miri"   -> "miri"
        "cloudfree_gemini" -> "gemini"
        "cloudy_hst"       -> "hst"
    """
    return sim_name.split("_", 1)[1]
