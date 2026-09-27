# native_input.py
import ctypes
import math

user32 = ctypes.windll.user32

# 1. Force Windows DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

# 2. Disable Windows Cursor Shadow Programmatically
SPI_SETCURSORS_SHADOW = 0x101B
SPIF_UPDATEINIFILE = 0x01
SPIF_SENDCHANGE = 0x02
try:
    # Disables the drop-shadow rendering effect under the cursor
    user32.SystemParametersInfoW(SPI_SETCURSORS_SHADOW, 0, ctypes.c_void_p(0), SPIF_UPDATEINIFILE | SPIF_SENDCHANGE)
except Exception:
    pass

# Get actual physical display pixel resolution
SCREEN_WIDTH = user32.GetSystemMetrics(0)
SCREEN_HEIGHT = user32.GetSystemMetrics(1)

# Mouse event constants
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

def set_mouse_pos(norm_x, norm_y):
    """
    Directly sets the true Windows OS cursor position without trails or ghosting.
    """
    if math.isnan(norm_x) or math.isinf(norm_x):
        norm_x = 0.5
    if math.isnan(norm_y) or math.isinf(norm_y):
        norm_y = 0.5

    norm_x = max(0.0, min(1.0, float(norm_x)))
    norm_y = max(0.0, min(1.0, float(norm_y)))

    pixel_x = int(norm_x * SCREEN_WIDTH)
    pixel_y = int(norm_y * SCREEN_HEIGHT)

    user32.SetCursorPos(pixel_x, pixel_y)

def mouse_down(right=False):
    flag = MOUSEEVENTF_RIGHTDOWN if right else MOUSEEVENTF_LEFTDOWN
    user32.mouse_event(flag, 0, 0, 0, 0)

def mouse_up(right=False):
    flag = MOUSEEVENTF_RIGHTUP if right else MOUSEEVENTF_LEFTUP
    user32.mouse_event(flag, 0, 0, 0, 0)

def mouse_scroll(clicks):
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(clicks * 120), 0)