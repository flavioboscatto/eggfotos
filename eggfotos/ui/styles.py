"""Tema escuro do EggFotos (destaque cor de gema)."""

ACCENT = "#F2B134"

STYLESHEET = f"""
QWidget {{
    background: #232328;
    color: #E6E6E6;
    font-size: 10pt;
}}
QMainWindow::separator {{ background: #2d2d33; width: 4px; }}
QMenuBar, QMenu {{ background: #1d1d21; }}
QMenu::item:selected, QMenuBar::item:selected {{ background: {ACCENT}; color: #1d1d21; }}
QToolBar {{ background: #1d1d21; border: none; spacing: 4px; padding: 3px; }}
QToolButton {{ padding: 4px 8px; border-radius: 4px; }}
QToolButton:hover {{ background: #34343b; }}
QToolButton:pressed, QToolButton:checked {{ background: {ACCENT}; color: #1d1d21; }}
QDockWidget::title {{ background: #1d1d21; padding: 6px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: #1d1d21; padding: 6px 10px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
QTabBar::tab:selected {{ background: #2d2d33; color: {ACCENT}; }}
QGroupBox {{ border: 1px solid #3a3a42; border-radius: 6px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {ACCENT}; }}
QPushButton {{ background: #34343b; border: 1px solid #44444c; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #777; border-color: #333; }}
QPushButton#primary {{ background: {ACCENT}; color: #1d1d21; border: none; font-weight: bold; }}
QPushButton#primary:disabled {{ background: #5a4b2a; color: #999; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #1b1b1f; border: 1px solid #44444c; border-radius: 4px; padding: 3px 6px;
}}
QListWidget {{ background: #1b1b1f; border: 1px solid #3a3a42; border-radius: 4px; }}
QListWidget::item:selected {{ background: #4a3d1f; color: {ACCENT}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: #44444c; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: #8a6a2a; border-radius: 2px; }}
QProgressBar {{ border: 1px solid #44444c; border-radius: 4px; text-align: center; background: #1b1b1f; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QStatusBar {{ background: #1d1d21; }}
QLabel#hint {{ color: #9a9aa3; font-size: 9pt; }}
QLabel#value {{ color: {ACCENT}; min-width: 34px; }}
"""
