import pytest

import paper_data_linking.data.download_text as dt


def test_download_and_parse():
    urls = [
        'https://link.springer.com/content/pdf/10.1007/s41116-021-00030-3.pdf?pdf=button',
        'https://export.arxiv.org/pdf/2104.04261',
    ]
    with dt.Downloader() as dl:
        txts = dl.download_and_parse(urls)
    print(txts)


@pytest.mark.parametrize("url,expected_exception", [
    ("https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014JA020272", UnboundLocalError),
    # ^ Selenium download will be attempted, but will fail because it's not a pdf download link.
    ("https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2014JA020272", None),
    ("https://articles.adsabs.harvard.edu/full/1995SoPh..162..357B?defaultprint=YES", None),
    ("https://stacks.iop.org/0004-637X/826/38/pdf", None),
    ("https://link.springer.com/content/pdf/10.1007/s41116-021-00030-3.pdf?pdf=button", None),
    ("https://arxiv.org/pdf/2211.11054.pdf", None),
])
def test_download_and_parse_inner(url, expected_exception):
    with dt.Downloader(headless=False) as dl:
        if expected_exception is None:
            txt = dl.download_and_parse_inner(url)
            assert isinstance(txt, str)
            # ^ may want better test to see if it's likely the actual content
        else:
            with pytest.raises(expected_exception):
                dl.download_with_selenium(url)