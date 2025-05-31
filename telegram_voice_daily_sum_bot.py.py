from __future__ import annotations

import datetime
import os
import sqlite3
import logging
import json

import resampy
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from telegram import Update, Voice, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.ext import CommandHandler

try:
    import openai
except ImportError:
    openai = None

# ────────────────────────────────────────────────────────────────────────────────
#  Utils
# ────────────────────────────────────────────────────────────────────────────────
class Utils:

    @staticmethod
    def now_utc() -> str:
        return datetime.datetime.now().replace(microsecond=0).isoformat()

    @staticmethod
    def ensure_parent(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def as_bool(value: str | bool | None) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def configure_logging(log_path: str):
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.FileHandler(log_path, encoding="utf‑8"),
                logging.StreamHandler(),
            ],
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)

# ────────────────────────────────────────────────────────────────────────────────
#  Database Layer
# ────────────────────────────────────────────────────────────────────────────────
class Database:
    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS voice_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_file_id TEXT UNIQUE NOT NULL,
        text        TEXT,
        summarized  TEXT,
        sent_at     TEXT,
        user_id     INTEGER,
        username    TEXT
    );"""

    def __init__(self, path: str):
        Utils.ensure_parent(Path(path))
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency
        self.conn.execute("PRAGMA busy_timeout = 10000;")
        self.conn.execute(self._CREATE_TABLE_SQL)
        self.conn.commit()
        logging.info("Connected to SQLite DB at %s", path)

    def save_message(
        self,
        telegram_file_id: str,
        text: str | None,
        summary: str | None,
        sent_at: str,
        user_id: int,
        username: str | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO voice_messages
                (telegram_file_id, text, summarized, sent_at, user_id, username)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(telegram_file_id) DO UPDATE SET
                text=excluded.text,
                summarized=excluded.summarized,
                sent_at=excluded.sent_at;
            """,
            (telegram_file_id, text, summary, sent_at, user_id, username),
        )
        self.conn.commit()
        logging.debug("Saved message %s", telegram_file_id)

    # Example of an accessor you might need later.
    def fetch_last_n(self, n: int = 10):
        cur = self.conn.execute(
            "SELECT * FROM voice_messages ORDER BY id DESC LIMIT ?", (n,)
        )
        return cur.fetchall()

    def fetch_summaries_by_date(self, date_str: str) -> list[tuple[str, str, str]]:
        cur = self.conn.execute(
            """
            SELECT sent_at, telegram_file_id, summarized
              FROM voice_messages
             WHERE sent_at LIKE ?
          ORDER BY sent_at ASC
            """,
            (f"{date_str}%",)
        )
        return cur.fetchall()

# ────────────────────────────────────────────────────────────────────────────────
#  LLM Handler
# ────────────────────────────────────────────────────────────────────────────────
class LLMHandler:
    def __init__(self, *, use_local: bool, cfg: Dict[str, str]):
        self.use_local = use_local
        self.cfg = cfg
        if not use_local and openai is None:
            raise RuntimeError(
                "OpenAI Python package is not installed but USE_LOCAL_LLM=False." )
        if not use_local:
            openai.api_key = cfg["OPENAI_API_KEY"]

    def summarize(self, text: str) -> str:
        deviation = int(self.cfg.get("SUMMARY_DEVIATION_PERCENT", 20))

        n = len(text)
        if n < 500:
            min_len, max_len = 150, 250
        else:
            min_len, max_len = max(150, n // 5), n // 3  # 5:1 и 3:1

        # сколько раз можно пробовать
        tries = int(self.cfg.get("SUMMARY_MAX_TRIES", 2))
        tries = max(1, tries)  # хотя бы одна

        def _generate() -> str:
            prompt = (
                "Сделай краткое резюме объёмом от "
                f"{min_len} до {max_len} символов (без кавычек, только текст) "
                "для следующего голосового сообщения:\n\n"
                f"{text}"
            )
            return (
                self._ollama_call(prompt)
                if self.use_local
                else self._openai_call(prompt)
            ).strip()

        for attempt in range(tries):
            summary = _generate()
            length = len(summary)
            if min_len * (1-(deviation/100)) <= length <= max_len * (1+(deviation/100)):
                break
            if attempt == tries - 1:
                # последняя попытка, примем как есть
                logging.warning(
                    "Summary length %d вне диапазона (%d‑%d) даже после %d попыток",
                    length, min_len, max_len, tries
                )

        return summary

    # ─── Private helpers ───────────────────────────────────────────────────────
    def _ollama_call(self, prompt: str) -> str:
        url = f"{self.cfg['OLLAMA_BASE_URL']}/api/generate"
        payload = {"model": self.cfg["OLLAMA_MODEL"], "prompt": prompt, "stream": False}
        logging.debug("Ollama request: %s", json.dumps(payload)[:200])
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        logging.info("Ollama summary len=%d", len(summary))
        return summary

    def _openai_call(self, prompt: str) -> str:
        logging.debug("OpenAI prompt len=%d", len(prompt))
        resp = openai.ChatCompletion.create(
            model=self.cfg["OPENAI_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        summary = resp.choices[0].message.content.strip()
        logging.info("OpenAI summary len=%d", len(summary))
        return summary

# ────────────────────────────────────────────────────────────────────────────────
#  Speech‑to‑Text
# ────────────────────────────────────────────────────────────────────────────────
class SpeechToText:
    SAMPLE_RATE = 16_000

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Vosk model not found: {model_path}")
        self.model = Model(model_path)
        logging.info("Loaded Vosk model from %s", model_path)

    def transcribe_ogg_bytes(self, ogg_bytes: bytes) -> str:
        """OGG ➜ WAV ➜ TEXT (returns transcription)."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg:
            ogg.write(ogg_bytes)
            ogg_path = Path(ogg.name)
        wav_path = ogg_path.with_suffix(".wav")
        self._ogg_to_wav(ogg_path, wav_path)
        text = self._wav_to_text(wav_path)
        # cleanup temp files
        ogg_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        return text

    def _ogg_to_wav(self, ogg_path: Path, wav_path: Path) -> None:
        # Читаем аудио
        data, sr = sf.read(str(ogg_path))

        # Приводим к моно
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Ресемплируем, если исходная частота != 16 kHz
        if sr != self.SAMPLE_RATE:
            data = resampy.resample(data, sr, self.SAMPLE_RATE)

        # Записываем WAV
        sf.write(str(wav_path), data, self.SAMPLE_RATE, subtype='PCM_16')

    def _wav_to_text(self, wav_path: Path) -> str:
        rec = KaldiRecognizer(self.model, self.SAMPLE_RATE)
        with open(wav_path, "rb") as wf:
            while chunk := wf.read(4000):
                rec.AcceptWaveform(chunk)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()

# ────────────────────────────────────────────────────────────────────────────────
#  Telegram Bot Wrapper
# ────────────────────────────────────────────────────────────────────────────────
class TelegramBot:
    def __init__(
        self,
        token: str,
        db: Database,
        stt: SpeechToText,
        llm: LLMHandler,
    ):
        self.db = db
        self.stt = stt
        self.llm = llm
        self.app = ApplicationBuilder().token(token).build()
        self.app.add_handler(
            MessageHandler(filters.VOICE & ~filters.COMMAND, self.on_voice)
        )
        self.app.add_handler(CommandHandler("sum", self.on_summary))
        logging.info("Telegram bot initialized – waiting for voice messages…")

    # ---------------------------------------------------------------------
    async def on_voice(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        assert isinstance(update.effective_message.voice, Voice)
        voice: Voice = update.effective_message.voice
        sent_at = Utils.now_utc()
        user = update.effective_user
        file_id = voice.file_id
        logging.info(
            "Voice received: file_id=%s | from=%s", file_id, user.username or user.id
        )
        tg_file = await ctx.bot.get_file(file_id)
        ogg_bytes = await tg_file.download_as_bytearray()

        # 1️⃣ Transcribe
        text = self.stt.transcribe_ogg_bytes(ogg_bytes)
        logging.info("Transcribed %d chars", len(text))
        logging.info(f"Text: {text}")

        # 2️⃣ Summarize
        summary = self.llm.summarize(text)

        # 3️⃣ Persist
        self.db.save_message(
            telegram_file_id=file_id,
            text=text,
            summary=summary,
            sent_at=sent_at,
            user_id=user.id,
            username=user.username,
        )

        # 4️⃣ Respond with summary
        await update.message.reply_text(
            f"#{file_id} \n📌 Резюме: {summary}", parse_mode=constants.ParseMode.HTML,
            reply_to_message_id=update.effective_message.message_id
        )

    async def on_summary(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.effective_message.text.strip()
        parts = text.split(maxsplit=1)

        if len(parts) > 1:
            date_str = parts[1]
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                await update.message.reply_text("❗️Неверный формат даты. Используйте YYYY-MM-DD")
                return
        else:
            tz = datetime.timezone(datetime.timedelta(hours=4))
            date_str = datetime.datetime.now(tz).date().isoformat()

        summaries = self.db.fetch_summaries_by_date(date_str)
        if not summaries:
            await update.message.reply_text(f"ℹ️ Нет резюме за {date_str}.")
            return

        lines = []
        for sent_at, telegram_file_id, summary in summaries:
            lines.append(f"{sent_at} — #{telegram_file_id}: {summary}\n")

        # Если очень много строк, можно разбить на части
        await update.message.reply_text("\n".join(lines))

    def run(self):
        self.app.run_polling()

# ────────────────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(dotenv_path='.env')

    Utils.configure_logging(os.getenv("LOG_FILE", "bot.log"))

    db = Database(os.getenv("DB_PATH", "bot.db"))

    stt = SpeechToText(model_path=os.getenv("VOSK_MODEL_PATH", "models/vosk‑small‑ru‑0.22"))

    llm = LLMHandler(
        use_local=Utils.as_bool(os.getenv("USE_LOCAL_LLM", "True")),
        cfg=os.environ,
    )

    bot = TelegramBot(
        token=os.environ["TG_BOT_TOKEN"],
        db=db,
        stt=stt,
        llm=llm,
    )

    bot.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        raise
