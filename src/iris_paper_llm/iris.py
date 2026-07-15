"""Canonical IRIS observable coverage used by classification."""

# De Pontieu et al. (2014), Tables 2 and 3:
# https://doi.org/10.1007/s11207-014-0485-y
IRIS_SPECTROGRAPH_WINDOWS_ANGSTROM = {
    "FUV1": (1331.7, 1358.4),
    "FUV2": (1389.0, 1407.0),
    "NUV": (2782.7, 2835.1),
}

# These are the standard channel names, not rectangular transmission ranges.
IRIS_SLIT_JAW_CHANNELS_ANGSTROM = (1330, 1400, 2796, 2832)
