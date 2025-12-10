import json
from datetime import datetime

from paper_data_linking import logger


class MetadataRecord:
    def __init__(
        self,
        bibcode: str,
        title: list[str],
        author: list[str],
        year: int,
        pub: str,
        links_data: list[dict],
        pubdate: str,
        doi: list[str],
        doctype: str,
        pdf_links: list[str] | None = None,
    ) -> None:
        self.bibcode = bibcode
        self.title = title  # Assuming title can be a list of strings (e.g., for multilingual titles)
        self.author = author  # Authors can be a list of strings
        self.year = int(year)
        self.pub = pub
        self.links_data = links_data
        standardized_pubdate = pubdate.replace("-00", "-01")  # Replace '00' day with '01'
        self.pubdate = datetime.strptime(standardized_pubdate, "%Y-%m-%d")  # NOQA: DTZ007
        self.doi = doi
        self.doctype = doctype
        self.pdf_links = pdf_links or []

    def to_dict(self):
        """
        Convert the object to a dictionary.
        """
        return {
            "bibcode": self.bibcode,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "pub": self.pub,
            "links_data": self.links_data,
            "pubdate": self.pubdate.strftime("%Y-%m-%d"),
            "doi": self.doi,
            "doctype": self.doctype,
            "pdf_links": self.pdf_links,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance from a dictionary.
        """
        links_data = [json.loads(link) if isinstance(link, str) else link for link in data.get("links_data", [])]
        if "author" not in data:
            logger.debug(f"Missing 'author' in data: {data}")
        return cls(
            bibcode=data["bibcode"],
            title=data["title"],
            author=data.get("author", ""),
            year=data["year"],
            pub=data["pub"],
            links_data=links_data,
            pubdate=data["pubdate"],
            doi=data.get("doi", []),
            doctype=data["doctype"],
            pdf_links=data.get("pdf_links", []),
        )


class BasicMetadataRecord:
    def __init__(
        self,
        bibcode: str,
        links_data: list[dict],
        pdf_links: list[str] | None = None,
    ) -> None:
        self.bibcode = bibcode
        self.links_data = links_data
        self.pdf_links = pdf_links or []

    def to_dict(self):
        """
        Convert the object to a dictionary.
        """
        return {
            "bibcode": self.bibcode,
            "links_data": self.links_data,
            "pdf_links": self.pdf_links,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance from a dictionary.
        """
        links_data = [json.loads(link) if isinstance(link, str) else link for link in data.get("links_data", [])]
        return cls(
            bibcode=data["bibcode"],
            links_data=links_data,
            pdf_links=data.get("pdf_links", []),
        )
