"""Telegram alerts (optional). Enable in config.yaml and put
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in ../.env - never in code."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


class TelegramAlerter:
    def __init__(self, enabled: bool):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = enabled and bool(self.token and self.chat_id)
        if enabled and not self.enabled:
            logger.warning("Telegram enabled in config but token/chat id missing in .env")

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10,
            ).raise_for_status()
        except Exception:
            logger.warning("Telegram alert failed (non-fatal)", exc_info=True)
