import json


def load_bibcodes(file_path):
    """
    Load bibcodes from a file.
    """
    with open(file_path) as file:
        return [line.strip() for line in file]


def load_search_results(file_path):
    """
    Load search results from a JSON file.
    """
    with open(file_path) as file:
        return json.load(file)


def compare_keys(iris_bibcodes, search_results):
    """
    Compare bibcodes from the IRIS list with those in the search results.
    """
    iris_bibcodes = set(iris_bibcodes)
    true_bibcodes = set()
    false_bibcodes = set()
    for bibcode, result in search_results.items():
        iris_classification = result.get("iris_classification")
        if iris_classification:
            if "YES" in result.get("iris_classification"):
                true_bibcodes.add(bibcode)
            else:
                false_bibcodes.add(bibcode)
        else:
            false_bibcodes.add(bibcode)
    missing_positive_bibcodes = true_bibcodes - iris_bibcodes
    false_positive_bibcodes = false_bibcodes.intersection(iris_bibcodes)
    found_positive_bibcodes = true_bibcodes.intersection(iris_bibcodes)
    unavailable_bibcodes = iris_bibcodes - true_bibcodes - false_bibcodes
    return (
        true_bibcodes,
        missing_positive_bibcodes,
        false_bibcodes,
        false_positive_bibcodes,
        unavailable_bibcodes,
        found_positive_bibcodes,
    )


if __name__ == "__main__":
    IRIS_LIST_BIBCODES_FILE = "iris_lib_bibcodes.txt"
    IRIS_SEARCH_RESULTS_FILE = "iris_search_results.json"
    IRIS_FAILED_DOWNLOADED = "failed_bibcodes.txt"
    ads_bibcodes = load_bibcodes(IRIS_LIST_BIBCODES_FILE)
    failed_downloads = load_bibcodes(IRIS_FAILED_DOWNLOADED)
    if failed_downloads:
        pass
    search_results = load_search_results(IRIS_SEARCH_RESULTS_FILE)
    missing_bibcodes_from_ads_based_on_query = set(ads_bibcodes) - set(search_results)
    (
        true_bibcodes,
        missing_positive_bibcodes,
        false_bibcodes,
        false_positive_bibcodes,
        unavailable_bibcodes,
        found_positive_bibcodes,
    ) = compare_keys(ads_bibcodes, search_results)
    print_per_bib = False
    if missing_positive_bibcodes:
        for _bibcode in sorted(missing_positive_bibcodes):
            pass
    else:
        pass
    if found_positive_bibcodes:
        pass
    else:
        pass
    if false_bibcodes:
        pass
    else:
        pass
    if false_positive_bibcodes:
        pass
    else:
        pass
    if unavailable_bibcodes:
        for _bibcode in sorted(unavailable_bibcodes):
            pass
    else:
        pass
