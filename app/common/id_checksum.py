_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_CODES = "10X98765432"


def id_card_checksum_valid(id_no: str) -> bool:
    """Validate 18-digit Chinese resident ID checksum."""
    if len(id_no) != 18:
        return False

    body = id_no[:17]
    if not body.isdigit():
        return False

    check_char = id_no[17].upper()
    if check_char not in "0123456789X":
        return False

    total = sum(int(body[i]) * _ID_WEIGHTS[i] for i in range(17))
    return _ID_CHECK_CODES[total % 11] == check_char
