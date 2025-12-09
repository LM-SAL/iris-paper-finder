import random
from typing import Literal
from dataclasses import field, dataclass

__all__ = ["build_headers"]

Platform = Literal["windows", "mac", "linux", "android", "ios"]
Browser = Literal["chrome", "firefox", "safari"]
_DEFAULT_ACCEPT_HTML = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
)
_HOP_BY_HOP = {"connection", "proxy-connection", "keep-alive", "transfer-encoding", "upgrade"}


@dataclass(frozen=True)
class HeaderSettings:
    locale: str = "en-US,en;q=0.9"
    mobile: bool = False
    platform: Platform = "linux"
    browser: Browser = "chrome"
    referer: str | None = None
    dnt: bool = False
    cache_control: str | None = None
    randomize_versions: bool = True
    extra: dict[str, str] = field(default_factory=dict)


def _rand(minor_low: int, minor_high: int) -> int:
    return random.randint(minor_low, minor_high)


def _chrome_ua(settings: HeaderSettings) -> str:
    base_major = 140
    base_major_high = 145
    major = _rand(base_major, base_major_high) if settings.randomize_versions else base_major_high
    minor = _rand(0, 1) if settings.randomize_versions else 0
    build = _rand(0, 9999) if settings.randomize_versions else 0
    patch = 0
    if settings.mobile:
        return (
            f"Mozilla/5.0 (Linux; Android 13; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.{minor}.{build}.{patch} Mobile Safari/537.36"
        )
    if settings.platform == "windows":
        plat = "Windows NT 10.0; Win64; x64"
    elif settings.platform == "mac":
        plat = "Macintosh; Intel Mac OS X 10_15_7"
    else:
        plat = "X11; Linux x86_64"
    return (
        f"Mozilla/5.0 ({plat}) AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.{minor}.{build}.{patch} Safari/537.36"
    )


def _firefox_ua(settings: HeaderSettings) -> str:
    gecko_major = _rand(120, 132) if settings.randomize_versions else 132
    if settings.mobile:
        return f"Mozilla/5.0 (Android 13; Mobile; rv:{gecko_major}.0) Gecko/{gecko_major}.0 Firefox/{gecko_major}.0"
    if settings.platform == "windows":
        plat = "Windows NT 10.0; Win64; x64"
    elif settings.platform == "mac":
        plat = "Macintosh; Intel Mac OS X 10.15"
    else:
        plat = "X11; Linux x86_64"
    return f"Mozilla/5.0 ({plat}; rv:{gecko_major}.0) Gecko/20100101 Firefox/{gecko_major}.0"


def _safari_ua(settings: HeaderSettings) -> str:
    webkit = "605.1.15"
    if settings.mobile or settings.platform in ("ios", "mac"):
        # iOS Safari UA pattern (also used by Mobile Safari on iPad/iPhone)
        return (
            f"Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/{webkit} "
            "(KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
        )
    return (
        f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/{webkit} "
        "(KHTML, like Gecko) Version/16.6 Safari/605.1.15"
    )


def _ua(settings: HeaderSettings) -> str:
    if settings.browser == "chrome":
        return _chrome_ua(settings)
    if settings.browser == "firefox":
        return _firefox_ua(settings)
    return _safari_ua(settings)


def build_headers(
    settings: HeaderSettings | None = None,
    accept: str = _DEFAULT_ACCEPT_HTML,
) -> dict[str, str]:
    s = settings or HeaderSettings()
    headers: dict[str, str] = {
        "User-Agent": _ua(s),
        "Accept": accept,
        "Accept-Language": s.locale,
        "Accept-Encoding": "gzip, deflate, br",
    }
    if s.referer:
        headers["Referer"] = s.referer
    if s.dnt:
        headers["DNT"] = "1"
    if s.cache_control:
        headers["Cache-Control"] = s.cache_control
    for k, v in s.extra.items():
        lk = k.lower()
        if lk not in _HOP_BY_HOP:
            headers[k] = v
    return headers


def with_referer(headers: dict[str, str], referer: str) -> dict[str, str]:
    h = dict(headers)
    h["Referer"] = referer
    return h


def rotate_user_agent(
    headers: dict[str, str],
    pool: list[HeaderSettings] | None = None,
) -> dict[str, str]:
    if not pool:
        pool = [
            HeaderSettings(browser="chrome", platform="windows"),
            HeaderSettings(browser="chrome", platform="mac"),
            HeaderSettings(browser="firefox", platform="linux"),
            HeaderSettings(browser="firefox", platform="windows"),
            HeaderSettings(browser="chrome", platform="linux"),
            HeaderSettings(browser="safari", platform="mac"),
            HeaderSettings(browser="chrome", platform="android", mobile=True),
            HeaderSettings(browser="safari", platform="ios", mobile=True),
        ]
    profile = random.choice(pool)
    h = dict(headers)
    h["User-Agent"] = _ua(profile)
    h["Accept-Language"] = profile.locale
    return h
