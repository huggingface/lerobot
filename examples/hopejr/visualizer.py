# Color gradient function (0-2024 scaled to 0-10)
def value_to_color(value):
    # Clamp the value between 0 and 2024
    value = max(0, min(2024, value))
    
    # Scale from [0..2024] to [0..10]
    scaled_value = (value / 2024) * 10
    
    # Green to Yellow (scaled_value 0..5), then Yellow to Red (scaled_value 5..10)
    if scaled_value <= 5:
        r = int(255 * (scaled_value / 5))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (scaled_value - 5) / 5))
    b = 0
    
    return (r, g, b)