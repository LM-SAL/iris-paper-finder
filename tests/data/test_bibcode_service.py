import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, mock_open, MagicMock
from paper_data_linking.data.bibcode_service import BibcodeService  # Make sure to adjust the import path accordingly


class TestBibcodeService(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(self.tmpdir.name)
        self.service = BibcodeService(
            api_token="TEST_TOKEN",
            bibcode_dir=self.tmpdir.name,
            resume_token_loc=tmpdir_path / 'resume_token.txt',
            library_id="TEST_LIBRARY"
        )

    @patch('builtins.open', mock_open(read_data="5"))
    def test_get_start_existing(self):
        self.assertEqual(self.service._get_start(), 5)

    def test_get_start_non_existing(self):
        self.assertEqual(self.service._get_start(), 0)

    @patch('builtins.open', mock_open(read_data="TEST_BIBCODE\nANOTHER_BIBCODE"))
    def test_load_bibcodes(self):
        bibcodes = self.service.load_bibcodes()
        self.assertEqual(bibcodes, ["TEST_BIBCODE", "ANOTHER_BIBCODE"])

    @patch('builtins.open', mock_open(read_data="TEST_BIBCODE\nANOTHER_BIBCODE"))
    @patch.object(Path, "iterdir")
    def test_load_bibcodes(self, mock_iterdir):
        # Mocking the directory to have two files
        mock_file_1 = MagicMock()
        mock_file_1.name = "file1.txt"
        mock_file_2 = MagicMock()
        mock_file_2.name = "file2.txt"
        mock_iterdir.return_value = [mock_file_1, mock_file_2]

        bibcodes = self.service.load_bibcodes()

        # Since we have two mocked files, and each file has the same content,
        # we expect the bibcodes list to have the content repeated twice
        expected_bibcodes = ["TEST_BIBCODE", "ANOTHER_BIBCODE", "TEST_BIBCODE", "ANOTHER_BIBCODE"]
        self.assertEqual(bibcodes, expected_bibcodes)


if __name__ == '__main__':
    unittest.main()
