import asyncio
from abc import ABC, abstractmethod
from time import sleep
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from .headers import build_headers


class Downloader(ABC):
    @abstractmethod
    async def download(self, url: str) -> bytes:
        pass

    def __aenter__(self):
        return self

    @abstractmethod
    def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncRequestsDownloader(Downloader):
    def __init__(self, headers=None, max_retries=2, delay=1.5) -> None:
        self.headers = headers or build_headers()
        self.max_retries = max_retries
        self.delay = delay

    async def download(self, url: str) -> bytes:
        retries = 0
        errors = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            while retries <= self.max_retries:
                try:
                    async with session.get(url, timeout=360) as response:
                        if response.status == 200:
                            return await response.read()
                        await asyncio.sleep(self.delay)
                        retries += 1
                except aiohttp.ClientError as e:
                    errors.append(e)
                    retries += 1
                    await asyncio.sleep(self.delay)
        msg = f"Failed to download from {url} after {self.max_retries} retries with error"
        raise Exception(msg) from errors[0] if errors else None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()


class AsyncSeleniumDownloader(Downloader):
    def __init__(self, headless=True, log_path=None) -> None:
        self.headless = headless
        self.log_path = log_path
        self.driver = self.init_webdriver()
        self.executor = ThreadPoolExecutor()

    def init_webdriver(self):
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
            options.add_argument("--headless")

        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--verbose")

        if self.log_path is not None:
            options.add_argument(f"--log-path={self.log_path}")

        options.add_experimental_option(
            "prefs",
            {
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            },
        )

        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    async def download(self, url: str, wait_time=5) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._download_blocking, url, wait_time)

    def _download_blocking(self, url: str, wait_time: int) -> bytes:
        with TemporaryDirectory() as tmpdir:
            self.driver.command_executor._commands["send_command"] = (
                "POST",
                "/session/$sessionId/chromium/send_command",
            )
            params = {"cmd": "Page.setDownloadBehavior", "params": {"behavior": "allow", "downloadPath": tmpdir}}
            self.driver.execute("send_command", params)
            self.driver.get(url)
            sleep(wait_time)
            for file in Path(tmpdir).iterdir():
                with open(file, "rb") as f:
                    return f.read()
        msg = f"Failed to download content from {url} using Selenium."
        raise Exception(msg)

    def close(self):
        self.driver.quit()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
