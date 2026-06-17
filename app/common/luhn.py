def luhn_check(card_no: str) -> bool:
    """Bank card Luhn checksum."""
    if not card_no.isdigit():
        return False

    total = 0
    reverse_digits = card_no[::-1]

    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0
