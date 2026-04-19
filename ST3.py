import re
import sys
import traceback
from pathlib import Path

import torch
import librosa
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt6.QtGui import QFont, QTextBlockFormat, QTextCursor, QFontDatabase
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout, 
    QSlider
)

from transformers import (
    AutoModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

TEXT_LOCAL_FOLDER_NAME = "gigaam-v3"
TEXT_MODEL_REVISION = "rnnt"

IPA_LOCAL_FOLDER_NAME = "ipa-whisper-medium"

TARGET_SR = 16000

IPA_CHUNK_SECONDS = 25
IPA_CHUNK_OVERLAP_SECONDS = 2


def get_base_path():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def get_text_model_path():
    return get_base_path() / TEXT_LOCAL_FOLDER_NAME


def get_ipa_model_path():
    return get_base_path() / IPA_LOCAL_FOLDER_NAME


def load_audio(path: str, target_sr: int = TARGET_SR):
    audio_array, sr = librosa.load(path, sr=target_sr, mono=True)
    audio_array = np.asarray(audio_array, dtype=np.float32)
    return audio_array, sr

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ipa_to_russian_transcription(ipa: str) -> str:
    text = ipa.strip()

    text = text.replace("‖", " ")
    text = text.replace("|", " ")
    text = text.replace("ˌ", "")
    text = text.replace(".", "")

    replacements = [
        ("t͡ɕ", "ч"),
        ("d͡ʑ", "дж"),
        ("d͡ʐ", "дж"),
        ("t͡s", "ц"),
        ("ɕː", "щ"),
        ("ʂː", "шш"),
        ("ʐː", "жж"),
        ("nʲ", "н'"),
        ("mʲ", "м'"),
        ("lʲ", "л'"),
        ("rʲ", "р'"),
        ("tʲ", "т'"),
        ("dʲ", "д'"),
        ("sʲ", "с'"),
        ("zʲ", "з'"),
        ("pʲ", "п'"),
        ("bʲ", "б'"),
        ("fʲ", "ф'"),
        ("vʲ", "в'"),
        ("kʲ", "к'"),
        ("gʲ", "г'"),
        ("xʲ", "х'"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    single_map = {
        "a": "а",
        "b": "б",
        "d": "д",
        "e": "э",
        "f": "ф",
        "g": "г",
        "h": "х",
        "i": "и",
        "j": "й",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "x": "х",
        "z": "з",


        "æ": "а",
        "ɑ": "а",
        "ɐ": "а",
        "ə": "э",
        "ɛ": "э",
        "ɨ": "ы",
        "ɪ": "и",
        "ʊ": "у",
        "ɔ": "о",
        "ɒ": "о",
        "ʏ": "ы",

        "ʒ": "ж",
        "ʌ": "о",
        "ʃ": "ш",
        "ɭ": "л",
        "ɫ": "л",
        "ʂ": "ш",
        "ʐ": "ж",
        "ɕ": "щ",
        "ʑ": "ж'",
        "ɲ": "н'",
        "ŋ": "нг",
        "ɡ": "г",

        "ʲ": "'",
        "ː": ":",
    }

    result = []
    for ch in text:
        result.append(single_map.get(ch, ch))

    text = "".join(result)

    text = text.replace("ь ", "ь ")
    text = normalize_spaces(text)

    return text


class TranscriptionWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str, str)
    finished_error = pyqtSignal(str)

    def __init__(self, audio_path: str, text_model, ipa_processor, ipa_model):
        super().__init__()
        self.audio_path = audio_path
        self.text_model = text_model
        self.ipa_processor = ipa_processor
        self.ipa_model = ipa_model

    def run(self):
        try:
            self.progress_changed.emit(5, "Проверка аудио...")
            audio_array, sampling_rate = load_audio(self.audio_path)

            self.progress_changed.emit(20, "Распознавание обычного текста...")
            text_result = self.generate_text()

            self.progress_changed.emit(65, "Построение IPA...")
            ipa_result = self.generate_ipa(audio_array, sampling_rate)

            self.progress_changed.emit(95, "Преобразование IPA в русскую транскрипцию...")
            russian_transcription = ipa_to_russian_transcription(ipa_result)

            self.progress_changed.emit(100, "Готово.")
            self.finished_ok.emit(text_result, ipa_result, russian_transcription)

        except Exception:
            self.finished_error.emit(traceback.format_exc())

    def generate_text(self):
        # Для локальной GigaAM-v3
        text = self.text_model.transcribe(self.audio_path)
        if isinstance(text, str):
            return text.strip()
        return str(text).strip()

    def generate_ipa(self, audio_array, sampling_rate):
        chunk_size = int(IPA_CHUNK_SECONDS * sampling_rate)
        overlap = int(IPA_CHUNK_OVERLAP_SECONDS * sampling_rate)
        step = max(1, chunk_size - overlap)

        total_len = len(audio_array)
        parts = []

        if total_len <= chunk_size:
            return self._generate_ipa_chunk(audio_array, sampling_rate)

        chunks = []
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk = audio_array[start:end]
            chunks.append(chunk)

            if end >= total_len:
                break

            start += step

        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks, start=1):
            progress = 65 + int((i / total_chunks) * 25)
            self.progress_changed.emit(progress, f"Построение IPA... {i}/{total_chunks}")

            chunk_text = self._generate_ipa_chunk(chunk, sampling_rate).strip()
            if chunk_text:
                parts.append(chunk_text)

        return normalize_spaces(" ".join(parts))

    def _generate_ipa_chunk(self, audio_array, sampling_rate):
        inputs = self.ipa_processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features

        attention_mask = torch.ones(
            input_features.shape[:2],
            dtype=torch.long,
            device=input_features.device,
        )

        with torch.no_grad():
            predicted_ids = self.ipa_model.generate(
                input_features,
                attention_mask=attention_mask,
            )

        ipa = self.ipa_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

        return ipa


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("STT")
        self.resize(1000, 900)

        self.text_model = None
        self.ipa_processor = None
        self.ipa_model = None
        self.worker = None

        self.current_audio_path = None
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.7)

        self.play_button = QPushButton("Пуск")
        self.pause_button = QPushButton("Пауза")
        self.recognize_button = QPushButton("Начать распознавание")

        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.recognize_button.setEnabled(False)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_label = QLabel("00:00 / 00:00")

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_label = QLabel("Громкость")

        self.status_label = QLabel("Загрузка моделей...")
        self.choose_button = QPushButton("Выбрать аудиофайл")
        self.choose_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.text_label = QLabel("Текст:")
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        self.ipa_label = QLabel("IPA:")
        self.ipa_output = QTextEdit()
        self.ipa_output.setReadOnly(True)

        self.rus_label = QLabel("Русская транскрипция:")
        self.rus_output = QTextEdit()
        self.rus_output.setReadOnly(True)

        self.setup_fonts()

        layout = QVBoxLayout()
        player_buttons_layout = QHBoxLayout()
        player_buttons_layout.addWidget(self.play_button)
        player_buttons_layout.addWidget(self.pause_button)
        player_buttons_layout.addWidget(self.recognize_button)

        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(self.time_slider)
        timeline_layout.addWidget(self.time_label)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(self.volume_slider)
        layout.addWidget(self.status_label)
        layout.addWidget(self.choose_button)
        layout.addWidget(self.progress_bar)
        layout.addLayout(player_buttons_layout)
        layout.addLayout(timeline_layout)
        layout.addLayout(volume_layout)
        layout.addWidget(self.text_label)
        layout.addWidget(self.text_output)
        layout.addWidget(self.ipa_label)
        layout.addWidget(self.ipa_output)
        layout.addWidget(self.rus_label)
        layout.addWidget(self.rus_output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.choose_button.clicked.connect(self.choose_file)

        self.play_button.clicked.connect(self.play_audio)
        self.pause_button.clicked.connect(self.pause_audio)
        self.recognize_button.clicked.connect(self.start_recognition)

        self.time_slider.sliderMoved.connect(self.set_position)
        self.volume_slider.valueChanged.connect(self.set_volume)

        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)

        self.load_models()

    def setup_fonts(self):
        text_font = QFont("Segoe UI", 11)
        self.text_output.setFont(text_font)
        self.rus_output.setFont(text_font)

        preferred_fonts = ["Charis SIL", "Doulos SIL", "Noto Serif", "Noto Sans", "Segoe UI"]
        available = set(QFontDatabase.families())

        chosen_font = "Segoe UI"
        for name in preferred_fonts:
            if name in available:
                chosen_font = name
                break

        ipa_font = QFont(chosen_font, 16)
        self.ipa_output.setFont(ipa_font)

        self.text_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.ipa_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.rus_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        self.ipa_output.setMinimumHeight(220)
        self.rus_output.setMinimumHeight(180)

    def load_models(self):
        try:
            text_model_path = get_text_model_path()
            ipa_model_path = get_ipa_model_path()

            if not text_model_path.exists():
                raise FileNotFoundError(
                    f"Локальная папка GigaAM не найдена:\n{text_model_path}\n\n"
                    f"Положи папку '{TEXT_LOCAL_FOLDER_NAME}' рядом с этим .py файлом."
                )

            if not ipa_model_path.exists():
                raise FileNotFoundError(
                    f"Локальная папка IPA модели не найдена:\n{ipa_model_path}\n\n"
                    f"Положи папку '{IPA_LOCAL_FOLDER_NAME}' рядом с этим .py файлом."
                )

            self.status_label.setText("Загрузка локальной GigaAM-v3...")
            QApplication.processEvents()

            self.text_model = AutoModel.from_pretrained(
                str(text_model_path),
                revision=TEXT_MODEL_REVISION,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            self.text_model = self.text_model.to("cpu").float()
            self.text_model.eval()

            self.progress_bar.setValue(35)
            QApplication.processEvents()

            self.status_label.setText("Загрузка локальной IPA модели...")
            QApplication.processEvents()

            self.ipa_processor = WhisperProcessor.from_pretrained(
                str(ipa_model_path),
                local_files_only=True,
            )
            self.ipa_model = WhisperForConditionalGeneration.from_pretrained(
                str(ipa_model_path),
                local_files_only=True,
            )
            self.ipa_model = self.ipa_model.to("cpu")
            self.ipa_model.eval()

            self.ipa_model.config.forced_decoder_ids = None
            self.ipa_model.config.suppress_tokens = []
            self.ipa_model.generation_config.forced_decoder_ids = None

            self.progress_bar.setValue(100)
            self.status_label.setText("Модели загружены. Выберите аудиофайл.")
            self.choose_button.setEnabled(True)

        except Exception as e:
            self.status_label.setText("Ошибка загрузки моделей.")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модели:\n{e}")

    def start_recognition(self):
        if not self.current_audio_path:
            QMessageBox.warning(self, "Внимание", "Сначала выберите аудиофайл.")
            return

        self.choose_button.setEnabled(False)
        self.recognize_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Обработка: {self.current_audio_path}")

        self.worker = TranscriptionWorker(
            audio_path=self.current_audio_path,
            text_model=self.text_model,
            ipa_processor=self.ipa_processor,
            ipa_model=self.ipa_model,
        )
        self.worker.progress_changed.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_success)
        self.worker.finished_error.connect(self.on_error)
        self.worker.start()

    def play_audio(self):
        self.player.play()

    def pause_audio(self):
        self.player.pause()

    def set_position(self, position: int):
        self.player.setPosition(position)

    def set_volume(self, value: int):
        self.audio_output.setVolume(value / 100.0)

    def update_position(self, position: int):
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(position)
        self.time_slider.blockSignals(False)

        duration = self.player.duration()
        self.time_label.setText(f"{self.format_time(position)} / {self.format_time(duration)}")

    def update_duration(self, duration: int):
        self.time_slider.setRange(0, duration)
        position = self.player.position()
        self.time_label.setText(f"{self.format_time(position)} / {self.format_time(duration)}")

    def format_time(self, ms: int) -> str:
        total_seconds = max(0, ms // 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл",
            str(get_base_path()),
            "Audio files (*.wav *.mp3 *.flac *.ogg *.m4a);;All files (*.*)",
        )

        if not file_path:
            return

        self.current_audio_path = file_path

        self.player.setSource(QUrl.fromLocalFile(file_path))

        self.text_output.clear()
        self.ipa_output.clear()
        self.rus_output.clear()

        self.progress_bar.setValue(0)
        self.status_label.setText(f"Выбран файл: {file_path}")

        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.recognize_button.setEnabled(True)

    def set_ipa_text(self, text: str):
        self.ipa_output.clear()
        self.ipa_output.setPlainText(text)

        cursor = self.ipa_output.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)

        block_format = QTextBlockFormat()
        block_format.setLineHeight(
            180.0,
            QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )

        cursor.mergeBlockFormat(block_format)
        cursor.clearSelection()
        self.ipa_output.setTextCursor(cursor)

    def on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_success(self, text_result: str, ipa_result: str, russian_transcription: str):
        self.text_output.setPlainText(text_result)
        self.set_ipa_text(ipa_result)
        self.rus_output.setPlainText(russian_transcription)

        self.status_label.setText("Готово.")
        self.progress_bar.setValue(100)
        self.choose_button.setEnabled(True)
        self.recognize_button.setEnabled(True)

    def on_error(self, error_text: str):
        self.status_label.setText("Ошибка во время обработки.")
        self.choose_button.setEnabled(True)
        self.recognize_button.setEnabled(True)
        self.progress_bar.setValue(0)

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Ошибка")
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setText("Произошла ошибка во время обработки.")
        dlg.setDetailedText(error_text)
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()