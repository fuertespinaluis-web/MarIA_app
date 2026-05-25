from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QTimer, Slot, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


class SplashScreen(QWidget):
    loading_finished = Signal()

    def __init__(self, video_path: Path, main_window: QWidget, ms: int = 6000):
        super().__init__(None)

        self.video_path = Path(video_path)
        self.main_window = main_window
        self.ms = ms
        self._finished = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setStyleSheet("background:black;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_widget = QVideoWidget(self)
        layout.addWidget(self.video_widget)

        try:
            self.video_widget.setAspectRatioMode(Qt.KeepAspectRatioByExpanding)
        except Exception:
            pass

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.0)

        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.errorOccurred.connect(self._on_error)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.finish)

        self._set_maximized_geometry()

    def _set_maximized_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.setGeometry(geo)

    def start(self) -> None:
        print("[Splash] video_path:", self.video_path)
        print("[Splash] exists:", self.video_path.exists())

        if not self.video_path.exists():
            self.finish()
            return

        self._set_maximized_geometry()
        self.show()
        self.raise_()
        self.activateWindow()

        self.player.setSource(QUrl.fromLocalFile(str(self.video_path)))
        self.player.play()
        self.timer.start(self.ms)

    @Slot()
    def finish(self) -> None:
        if self._finished:
            return

        self._finished = True

        self.timer.stop()
        self.player.stop()
        self.close()

        self.main_window.setWindowOpacity(1.0)
        self.main_window.showMaximized()
        self.main_window.raise_()
        self.main_window.activateWindow()

        self.loading_finished.emit()

    @Slot()
    def _on_error(self, *args) -> None:
        print("[Splash] playback error")
        self.finish()

    @Slot()
    def _on_media_status_changed(self, status) -> None:
        print("[Splash] media status:", status)
        if status == QMediaPlayer.EndOfMedia:
            self.finish()


def show_splash_then(main_window: QWidget, ms: int = 6000) -> None:
    base_dir = Path(__file__).resolve().parent.parent

    candidates = [
        base_dir / "assets" / "Video Project.mp4",
        base_dir / "assets" / "splash.mp4",
        base_dir / "assets" / "maria_splash.mp4",
    ]

    video_path = None
    for p in candidates:
        print("[Splash] checking:", p)
        if p.exists():
            video_path = p
            break

    if video_path is None:
        print("[Splash] no video found")
        main_window.setWindowOpacity(1.0)
        main_window.showMaximized()
        return

    splash = SplashScreen(video_path=video_path, main_window=main_window, ms=ms)
    main_window._splash_ref = splash
    splash.start()
