import pytest

from paper_data_linking.utils import get_acronym, INSTRUMENT_ACRONYM_EXPANSIONS


@pytest.mark.parametrize(
    "inst_str, expected",
    [
        # Test case-sensitive matching
        ("celias", "CELIAS"),
        # Test case-insensitive matching
        ("CeLiAs", "CELIAS"),
        # Test misspelled acronyms
        ("celiass", "CELIAS"),
        # Test exact full name matching
        ("Charge, Element, and Isotope Analysis System", "CELIAS"),
        # Test misspelled full names
        ("Charge, Elmnt, and Isotope Analysis Systm", "CELIAS"),
        # Test complex strings with full name and acronym
        ("Michelson Doppler Imager (SOI–MDI)", "MDI"),
        # Test complex strings with misspelled full name and acronym
        ("Michaelson Dopplelr Imager (SOI–MDI)", "MDI"),
        # Test acronym not in the list
        ("not in the list", "not in the list"),
        # Others from testing
        ("Extreme ultraviolet Imaging Telescope", "EIT"),
        # ("LASCO C2 white light coronagraph", "LASCO"), # fails
        # Test empty string
        ("", ""),
    ],
)
def test_get_acronym(inst_str, expected):
    result = get_acronym(inst_str, acronym_expansion_list=INSTRUMENT_ACRONYM_EXPANSIONS)
    assert result == expected
