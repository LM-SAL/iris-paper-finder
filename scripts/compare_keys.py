import json

def load_bibcodes(file_path):
    """
    Load bibcodes from a file.
    """
    with open(file_path, "r") as file:
        bibcodes = [line.strip() for line in file]
    return bibcodes

def load_search_results(file_path):
    """
    Load search results from a JSON file.
    """
    with open(file_path, "r") as file:
        search_results = json.load(file)
    return search_results

def compare_keys(iris_bibcodes, search_results):
    """
    Compare bibcodes from the IRIS list with those in the search results.
    """
    iris_bibcodes = set(iris_bibcodes)
    true_bibcodes = set()
    false_bibcodes = set()
    print("Comparing IRIS bibcodes with search results...")
    print("")
    for bibcode, result in search_results.items():
        iris_classification = result.get("iris_classification")
        if iris_classification:
            if "YES" in result.get("iris_classification"):
                true_bibcodes.add(bibcode)
            else:
                #print(f"Bibcode {bibcode} not classified as YES in search results.")
                false_bibcodes.add(bibcode)
        else:
            print(f"Bibcode {bibcode} has no iris_classification in search results.")
            print("This bibcode will be added to the false.")
            false_bibcodes.add(bibcode)
    missing_positive_bibcodes = true_bibcodes - iris_bibcodes
    false_positive_bibcodes = false_bibcodes.intersection(iris_bibcodes)
    found_positive_bibcodes = true_bibcodes.intersection(iris_bibcodes)
    unavailable_bibcodes = iris_bibcodes - true_bibcodes - false_bibcodes
    return true_bibcodes, missing_positive_bibcodes, false_bibcodes, false_positive_bibcodes, unavailable_bibcodes, found_positive_bibcodes


if __name__ == "__main__":
    IRIS_LIST_BIBCODES_FILE = "iris_lib_bibcodes.txt"
    IRIS_SEARCH_RESULTS_FILE = "iris_search_results.json"
    IRIS_FAILED_DOWNLOADED = "failed_bibcodes.txt"
    ads_bibcodes = load_bibcodes(IRIS_LIST_BIBCODES_FILE)
    failed_downloads = load_bibcodes(IRIS_FAILED_DOWNLOADED)
    if failed_downloads:
        print("Of the failed downloads, the number on the IRIS list is", len(set(failed_downloads).intersection(set(ads_bibcodes))))
    search_results = load_search_results(IRIS_SEARCH_RESULTS_FILE)
    missing_bibcodes_from_ads_based_on_query = set(ads_bibcodes) - set(search_results)
    print(f"{missing_bibcodes_from_ads_based_on_query}")
    print(f"{len(missing_bibcodes_from_ads_based_on_query)}")
    true_bibcodes, missing_positive_bibcodes, false_bibcodes, false_positive_bibcodes, unavailable_bibcodes, found_positive_bibcodes = compare_keys(ads_bibcodes, search_results)
    print_per_bib = False
    print("In total, there are", len(search_results), "bibcodes classified by the LLM.")
    print("In total, there are", len(ads_bibcodes), "bibcodes in the ADS list.")
    print("In total, there are", len(true_bibcodes), "bibcodes classified as positive by the LLM.")
    print("In total, there are", len(false_bibcodes), "bibcodes classified as negative by the LLM.")
    print("Comparison results:")
    if missing_positive_bibcodes:
        print("Missing positive bibcodes:")
        for bibcode in sorted(missing_positive_bibcodes):
            print(bibcode)
        print("There are", len(missing_positive_bibcodes), "classified positive bibcodes missing from the ADS list.")
    else:
        print("No missing positive bibcodes found.")
    print("")
    if found_positive_bibcodes:
        print("In total, there are", len(found_positive_bibcodes), "classified positive bibcodes on the ADS list.")
    else:
        print("No found positive bibcodes found.")
    if false_bibcodes:
        print("False bibcodes:")
        print("There are", len(false_bibcodes), "classified negative bibcodes.")
    else:
        print("No false bibcodes found.")
    print("")
    if false_positive_bibcodes:
        print("False positive bibcodes:")
        print("There are", len(false_positive_bibcodes), "classified negative bibcodes on the ADS list.")
    else:
        print("No false positive bibcodes found.")
    print("")
    if unavailable_bibcodes:
        print("Unavailable bibcodes from search results:")
        #for bibcode in sorted(unavailable_bibcodes):
        #    print(bibcode)
        print("There are", len(unavailable_bibcodes), "bibcodes unable to be classified by the LLM.")
    else:
        print("No unavailable bibcodes from search results.")
    print("")
    print("Comparison complete.")