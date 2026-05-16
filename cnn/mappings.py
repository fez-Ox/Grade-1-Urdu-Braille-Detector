# 6-dot Braille logic


def pattern_to_int(pattern: str) -> int:
    if len(pattern) != 6:
        raise ValueError("Pattern must be exactly 6 characters long.")

    # Reverse the string because Unicode Braille uses LSB for Dot 1
    # Dot 1: bit 0 (val 1)
    # Dot 2: bit 1 (val 2)
    # ...
    # Dot 6: bit 5 (val 32)
    return int(pattern[::-1], 2)


def int_to_pattern(class_idx: int) -> str:
    if not (0 <= class_idx <= 63):
        raise ValueError("Class index must be between 0 and 63.")

    # 06b gives '000001' for 1, which we reverse to get '100000'
    return f"{class_idx:06b}"[::-1]


def int_to_unicode_braille(class_idx: int) -> str:
    if not (0 <= class_idx <= 63):
        raise ValueError("Class index must be between 0 and 63.")

    return chr(0x2800 + class_idx)
