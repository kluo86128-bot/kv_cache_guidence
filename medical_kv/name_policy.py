"""One explicit, idempotent transformation for Synthea's synthetic name suffixes."""
import re

NAME_FORMAT = 'natural_no_numeric_suffix_v1'
NAME_KEYS = ('FIRST', 'MIDDLE', 'LAST')


def natural_component(value):
    # Do not remove digits from dates, addresses, record IDs or diagnoses.
    return re.sub(r'[0-9]+$', '', value)


def natural_patient(patient):
    output = dict(patient)
    for key in (*NAME_KEYS, 'MAIDEN'):
        if key in output:
            output[key] = natural_component(output[key])
    return output


def has_name_digits(value):
    return any('0' <= ch <= '9' for ch in value)
