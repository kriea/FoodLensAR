import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

def button_click():
    print("Button clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Basic Window with Button")
    window.showFullScreen()

    button = QPushButton("Click Me!", window)
    button.setMaximumSize(100,100)
    button.clicked.connect(button_click)

    window.setCentralWidget(button)
    window.show()

    sys.exit(app.exec_())

