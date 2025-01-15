import pytest

import paper_data_linking.data.transform_urls as tr

testdata = [
    ("https://articles.adsabs.harvard.edu/full/1995SoPh..162..357B",
     "https://articles.adsabs.harvard.edu/full/1995SoPh..162..357B?defaultprint=YES"),
    ("https://articles.adsabs.harvard.edu/full/1995SoPh..162..357B?defaultprint=YES",
     "https://articles.adsabs.harvard.edu/full/1995SoPh..162..357B?defaultprint=YES"),
    ("https://arxiv.org/abs/1409.1422", "https://export.arxiv.org/pdf/1409.1422"),
    ("https://www.aanda.org/component/article?access=bibcode&bibcode=&bibcode=2004A%2526A...427..755GFUL",
     "https://www.aanda.org/articles/aa/pdf/2004/44/aa0131-04.pdf"),
    ("https://arxiv.org/abs/astro-ph/0603361", "https://export.arxiv.org/pdf/astro-ph/0603361"),
    ("https://angeo.copernicus.org/articles/26/213/2008/",
     "https://angeo.copernicus.org/articles/26/213/2008/angeo-26-213-2008.pdf"),
    ("https://www.degruyter.com/document/doi/10.1515/astro-2022-0023/html",
     "https://www.degruyter.com/document/doi/10.1515/astro-2022-0023/pdf"),
    ("https://geoscienceletters.springeropen.com/articles/10.1186/s40562-018-0103-1",
     "https://geoscienceletters.springeropen.com/counter/pdf/10.1186/s40562-018-0103-1.pdf"),
    ("https://onlinelibrary.wiley.com/doi/abs/10.1002/2014JA020272",
     "https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2014JA020272"),
    ("https://onlinelibrary.wiley.com/doi/10.1002/2013SW001024",
     "https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2013SW001024")
]


@pytest.mark.parametrize("url,expected_transformed_url", testdata)
def test_transform_pdf_url(url, expected_transformed_url):
    transformed_url = tr.transform_pdf_url(url)
    assert transformed_url == expected_transformed_url


testrecords = [
    {
        "bibcode": "2021LRSP...18....4T",
        "author": ["Temmer, Manuela"],
        "id": "20303174",
        "links_data": [
            {
                "access": "open",
                "instances": "",
                "title": "",
                "type": "preprint",
                "url": "http://arxiv.org/abs/2104.04261",
            },
            {
                "access": "open",
                "instances": "",
                "title": "",
                "type": "electr",
                "url": "https://doi.org/10.1007%2Fs41116-021-00030-3",
            }
        ],
        "pub": "Living Reviews in Solar Physics",
        "title": ["Space weather: the solar perspective"],
        "year": "2021"
    },
    {
        "bibcode": "1998AnGeo..16....1B",
        "author": ["Bothmer, V.", "Schwenn, R."],
        "id": "2064585",
        "links_data": [
            {
                "access": "open",
                "instances": "",
                "title": "",
                "type": "pdf",
                "url": "https://angeo.copernicus.org/articles/16/1/1998/angeo-16-1-1998.pdf"
            },
            {
                "access": "open",
                "instances": "",
                "title": "",
                "type": "pdf",
                "url": "https://www.ann-geophys.net/16/1/1998/angeo-16-1-1998.pdf"
            },
            {
                "access": "open",
                "instances": "",
                "title": "",
                "type": "electr",
                "url": "http://www.ann-geophys.net/16/1/1998/angeo-16-1-1998.html"
            }
        ],
        "pub": "Annales Geophysicae",
        "title": ["The structure and origin of magnetic clouds in the solar wind"],
        "year": "1998"
    },
]


@pytest.mark.parametrize("record", testrecords)
def test_get_pdf_urls(record):
    urls = tr.get_pdf_urls(record)
    print(urls)


def test_get_redirected_url():
    # url = "https://doi.org/10.1007%2Fs41116-021-00030-3"
    url = "https://onlinelibrary.wiley.com/doi/abs/10.1002/2014JA020272"
    redirected_url = tr.get_redirected_url(url)
    turl = tr.transform_pdf_url(redirected_url)
    print(turl)
