# windows-ai-agent-toolset-v2.1
# Single-file Windows 11 "computer-use" agent runner for LM Studio (OpenAI-compatible).
#
# Key upgrades vs v2:
# - Much faster screenshot capture: GDI StretchBlt to target size (no pure-Python Lanczos resize).
# - Cursor drawn AFTER scaling (sharper cursor for the VLM).
# - Virtual screen (multi-monitor) aware coordinate mapping.
# - Mouse click + wheel use SendInput (more reliable than mouse_event).
# - Robust WinAPI error handling + DPI-awareness fallback.
# - Scenario tool disabling when schema marks a tool "UNAVAILABLE".
# - Keep last K screenshots in conversation (default 1) rather than always deleting all.

from __future__ import annotations

import ctypes
import json
import time
import base64
import urllib.request
import urllib.error
import struct
import zlib
import logging
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Configuration (override via CLI flags where available)
# ----------------------------

LM_STUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-2b-instruct"

TIMEOUT = 240
MAX_STEPS = 20
STEP_DELAY = 0.4
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# Model image size (keep aligned with your tests)
TARGET_W = 1344
TARGET_H = 756

# Dumps
DUMP_SCREENSHOTS = True
DUMP_DIR = "dumps"
DUMP_PREFIX = "dump_screen_"
DUMP_START = 1

# Conversation hygiene
KEEP_LAST_SCREENSHOTS = 1  # keep last N user image messages in context

# Logging
VERBOSE_LOG_CONVERSATION = True  # set False to reduce log spam and minor overhead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

# ----------------------------
# Scenario loading (from test_scenarios.txt)
# ----------------------------

def load_scenario(filename: str, scenario_num: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Load a specific scenario from the test scenarios file.
    Notes:
      - Supports SHARED_SYSTEM_PROMPT block (as in your test_scenarios.txt).
      - Expects TOOL_SCHEMA JSON to be on the same line as 'TOOLS_SCHEMA:'.
        (If you later wrap that JSON, extend this to read until balanced braces.)
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        shared_system_prompt: Optional[str] = None
        if "=== SHARED_SYSTEM_PROMPT ===" in content:
            parts = content.split("=== SHARED_SYSTEM_PROMPT ===", 1)
            shared_block = parts[1].split("=== SCENARIO", 1)[0]
            shared_system_prompt = shared_block.strip()

        scenarios = content.split("=== SCENARIO ")
        max_scen = len(scenarios) - 1
        if scenario_num < 1 or scenario_num > max_scen:
            log.error("Invalid scenario number %d. Available: 1-%d", scenario_num, max_scen)
            sys.exit(1)

        scenario_text = scenarios[scenario_num]
        lines = scenario_text.strip().split("\n")

        system_prompt = shared_system_prompt or ""
        task_prompt: Optional[str] = None
        tools_schema: Optional[List[Dict[str, Any]]] = None

        for raw in lines:
            line = raw.strip()
            if line.startswith("SYSTEM_PROMPT:"):
                system_prompt = line[len("SYSTEM_PROMPT:"):].strip()
            elif line.startswith("TASK_PROMPT:"):
                task_prompt = line[len("TASK_PROMPT:"):].strip()
            elif line.startswith("TOOLS_SCHEMA:"):
                tools_schema_str = line[len("TOOLS_SCHEMA:"):].strip()
                tools_schema = json.loads(tools_schema_str)

        if not system_prompt or not task_prompt or not tools_schema:
            log.error("Failed to parse scenario %d (missing prompt or schema).", scenario_num)
            sys.exit(1)

        log.info("Loaded scenario %d from %s", scenario_num, filename)
        return system_prompt, task_prompt, tools_schema

    except FileNotFoundError:
        log.error("Scenario file not found: %s", filename)
        sys.exit(1)
    except Exception as e:
        log.error("Error loading scenario: %s", e)
        sys.exit(1)

# ----------------------------
# Win32 API Setup
# ----------------------------

if os.name != "nt":
    raise OSError("This script requires Windows (uses Win32 APIs via ctypes).")

from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

# Optional libraries for DPI fallback
try:
    shcore = ctypes.WinDLL("shcore", use_last_error=True)
except Exception:
    shcore = None

# Some wintypes missing in older Python builds
if not hasattr(wintypes, "HCURSOR"):
    wintypes.HCURSOR = wintypes.HANDLE
if not hasattr(wintypes, "HBITMAP"):
    wintypes.HBITMAP = wintypes.HANDLE
if not hasattr(wintypes, "HICON"):
    wintypes.HICON = wintypes.HANDLE

# DPI awareness constants
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

# Virtual screen metrics
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Cursor / icon structures
class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HCURSOR),
        ("ptScreenPos", POINT),
    ]

class ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", wintypes.HBITMAP),
        ("hbmColor", wintypes.HBITMAP),
    ]

CURSOR_SHOWING = 0x00000001
DI_NORMAL = 0x0003

# DIB structs
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

BI_RGB = 0
DIB_RGB_COLORS = 0

# SendInput structs
try:
    ULONG_PTR = wintypes.ULONG_PTR
except AttributeError:
    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class INPUT_I(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_I)]

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800

# Stretch modes
HALFTONE = 4
SRCCOPY = 0x00CC0020

# Prototypes (best-effort; guard missing APIs)
user32.GetSystemMetrics.argtypes = [wintypes.INT]
user32.GetSystemMetrics.restype = wintypes.INT

user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC

user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = wintypes.INT

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC

gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ

gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL

gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD
]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP

gdi32.StretchBlt.argtypes = [
    wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
    wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
    wintypes.DWORD
]
gdi32.StretchBlt.restype = wintypes.BOOL

gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, wintypes.INT]
gdi32.SetStretchBltMode.restype = wintypes.INT

# SetBrushOrgEx is optional
if hasattr(gdi32, "SetBrushOrgEx"):
    gdi32.SetBrushOrgEx.argtypes = [wintypes.HDC, wintypes.INT, wintypes.INT, ctypes.POINTER(POINT)]
    gdi32.SetBrushOrgEx.restype = wintypes.BOOL

user32.SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
user32.SetCursorPos.restype = wintypes.BOOL

user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL

user32.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
user32.GetCursorInfo.restype = wintypes.BOOL

user32.GetIconInfo.argtypes = [wintypes.HICON, ctypes.POINTER(ICONINFO)]
user32.GetIconInfo.restype = wintypes.BOOL

user32.DrawIconEx.argtypes = [
    wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.HICON,
    wintypes.INT, wintypes.INT, wintypes.UINT, wintypes.HBRUSH, wintypes.UINT
]
user32.DrawIconEx.restype = wintypes.BOOL

user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

# DPI APIs (optional)
_has_SetProcessDpiAwarenessContext = hasattr(user32, "SetProcessDpiAwarenessContext")
if _has_SetProcessDpiAwarenessContext:
    user32.SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
    user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL

_has_SetProcessDPIAware = hasattr(user32, "SetProcessDPIAware")
if _has_SetProcessDPIAware:
    user32.SetProcessDPIAware.argtypes = []
    user32.SetProcessDPIAware.restype = wintypes.BOOL

if shcore and hasattr(shcore, "SetProcessDpiAwareness"):
    shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
    shcore.SetProcessDpiAwareness.restype = ctypes.c_long

# ----------------------------
# DPI + virtual screen helpers
# ----------------------------

def _dpi_aware() -> None:
    """
    Try to set per-monitor DPI awareness v2.
    Fallbacks: shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE=2),
               user32.SetProcessDPIAware().
    """
    # Best: SetProcessDpiAwarenessContext
    try:
        if _has_SetProcessDpiAwarenessContext:
            ok = user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            if ok:
                return
    except Exception:
        pass

    # Fallback: shcore.SetProcessDpiAwareness
    try:
        if shcore and hasattr(shcore, "SetProcessDpiAwareness"):
            # 2 = PROCESS_PER_MONITOR_DPI_AWARE
            shcore.SetProcessDpiAwareness(2)
            return
    except Exception:
        pass

    # Old fallback: SetProcessDPIAware
    try:
        if _has_SetProcessDPIAware:
            user32.SetProcessDPIAware()
    except Exception:
        pass

@dataclass
class VirtualScreen:
    left: int
    top: int
    width: int
    height: int

def _get_virtual_screen() -> VirtualScreen:
    _dpi_aware()
    left = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    top = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    w = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    h = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    # Fallback sane defaults
    if w <= 0 or h <= 0:
        left, top, w, h = 0, 0, 1920, 1080
    return VirtualScreen(left, top, w, h)

def _get_cursor_pos() -> Tuple[int, int]:
    p = POINT()
    if not user32.GetCursorPos(ctypes.byref(p)):
        return 0, 0
    return int(p.x), int(p.y)

def _cursor_pos_norm(vs: VirtualScreen) -> Tuple[int, int, int, int]:
    cx, cy = _get_cursor_pos()
    if vs.width <= 1 or vs.height <= 1:
        return cx, cy, 0, 0
    xn = int(round(((cx - vs.left) / (vs.width - 1)) * 1000.0))
    yn = int(round(((cy - vs.top) / (vs.height - 1)) * 1000.0))
    xn = 0 if xn < 0 else 1000 if xn > 1000 else xn
    yn = 0 if yn < 0 else 1000 if yn > 1000 else yn
    return cx, cy, xn, yn

def _norm_to_screen_px(xn: float, yn: float, vs: VirtualScreen) -> Tuple[int, int]:
    xn = 0.0 if xn < 0 else 1000.0 if xn > 1000 else xn
    yn = 0.0 if yn < 0 else 1000.0 if yn > 1000 else yn
    x = vs.left + int(round((xn / 1000.0) * (vs.width - 1)))
    y = vs.top + int(round((yn / 1000.0) * (vs.height - 1)))
    return x, y

# ----------------------------
# Screenshot capture (fast): StretchBlt to TARGET_W/H, then draw cursor on scaled image
# ----------------------------

def _raise_last_error(msg: str) -> None:
    err = ctypes.get_last_error()
    raise RuntimeError(f"{msg} (winerr={err})")

def _draw_cursor_on_scaled_dc(hdc_mem: int, vs: VirtualScreen, dst_w: int, dst_h: int) -> bool:
    ci = CURSORINFO()
    ci.cbSize = ctypes.sizeof(CURSORINFO)
    if not user32.GetCursorInfo(ctypes.byref(ci)):
        return False
    if not (ci.flags & CURSOR_SHOWING):
        return False

    ii = ICONINFO()
    if not user32.GetIconInfo(ci.hCursor, ctypes.byref(ii)):
        return False

    try:
        # Cursor top-left in *virtual screen* pixels after hotspot adjustment
        cur_x = int(ci.ptScreenPos.x) - int(ii.xHotspot)
        cur_y = int(ci.ptScreenPos.y) - int(ii.yHotspot)

        rel_x = cur_x - vs.left
        rel_y = cur_y - vs.top

        # scale to destination bitmap coordinates
        dx = int(round(rel_x * (dst_w / float(vs.width))))
        dy = int(round(rel_y * (dst_h / float(vs.height))))

        ok = user32.DrawIconEx(hdc_mem, dx, dy, ci.hCursor, 0, 0, 0, None, DI_NORMAL)
        return bool(ok)
    finally:
        if ii.hbmMask:
            gdi32.DeleteObject(ii.hbmMask)
        if ii.hbmColor:
            gdi32.DeleteObject(ii.hbmColor)

def _png_from_rgb24(rgb: bytes, w: int, h: int) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    row = w * 3
    raw = bytearray((row + 1) * h)
    for y in range(h):
        raw[y * (row + 1)] = 0
        off = y * row
        raw[y * (row + 1) + 1: y * (row + 1) + 1 + row] = rgb[off:off + row]

    comp = zlib.compress(bytes(raw), 6)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")

def take_screenshot_png() -> Tuple[bytes, VirtualScreen]:
    """
    Capture the *virtual screen* downscaled directly to TARGET_W x TARGET_H.
    Cursor is drawn AFTER scaling for better visibility and less blur.
    Returns (png_bytes, virtual_screen_info).
    """
    _dpi_aware()
    vs = _get_virtual_screen()

    hdc_screen = user32.GetDC(None)
    if not hdc_screen:
        _raise_last_error("GetDC(None) failed")

    hdc_mem = None
    hbmp = None
    old = None
    bits = ctypes.c_void_p()

    try:
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        if not hdc_mem:
            _raise_last_error("CreateCompatibleDC failed")

        bmi = BITMAPINFO()
        ctypes.memset(ctypes.byref(bmi), 0, ctypes.sizeof(bmi))
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = TARGET_W
        bmi.bmiHeader.biHeight = -TARGET_H  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32       # BGRA
        bmi.bmiHeader.biCompression = BI_RGB

        hbmp = gdi32.CreateDIBSection(hdc_mem, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), 0, 0)
        if not hbmp or not bits.value:
            _raise_last_error("CreateDIBSection failed")

        old = gdi32.SelectObject(hdc_mem, hbmp)
        if not old:
            _raise_last_error("SelectObject failed")

        gdi32.SetStretchBltMode(hdc_mem, HALFTONE)
        if hasattr(gdi32, "SetBrushOrgEx"):
            pt = POINT()
            gdi32.SetBrushOrgEx(hdc_mem, 0, 0, ctypes.byref(pt))

        ok = gdi32.StretchBlt(
            hdc_mem, 0, 0, TARGET_W, TARGET_H,
            hdc_screen, vs.left, vs.top, vs.width, vs.height,
            SRCCOPY
        )
        if not ok:
            _raise_last_error("StretchBlt failed")

        _draw_cursor_on_scaled_dc(hdc_mem, vs, TARGET_W, TARGET_H)

        # Read BGRA buffer
        size = TARGET_W * TARGET_H * 4
        buf = (ctypes.c_ubyte * size).from_address(bits.value)
        bgra = bytes(buf)

        # Convert BGRA -> RGB (no alpha)
        rgb = bytearray(TARGET_W * TARGET_H * 3)
        j = 0
        for i in range(0, len(bgra), 4):
            b = bgra[i]
            g = bgra[i + 1]
            r = bgra[i + 2]
            rgb[j] = r
            rgb[j + 1] = g
            rgb[j + 2] = b
            j += 3

        return _png_from_rgb24(bytes(rgb), TARGET_W, TARGET_H), vs

    finally:
        try:
            if hdc_mem and old:
                gdi32.SelectObject(hdc_mem, old)
        except Exception:
            pass
        try:
            if hbmp:
                gdi32.DeleteObject(hbmp)
        except Exception:
            pass
        try:
            if hdc_mem:
                gdi32.DeleteDC(hdc_mem)
        except Exception:
            pass
        try:
            user32.ReleaseDC(None, hdc_screen)
        except Exception:
            pass

# ----------------------------
# LM Studio request
# ----------------------------

def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LM_STUDIO_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTPError {e.code}: {body[:500]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e}")

# ----------------------------
# Tool arg parsing
# ----------------------------

def _parse_norm_xy(arg_str: Any) -> Tuple[float, float]:
    try:
        a = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
    except Exception:
        a = {}
    if isinstance(a, dict):
        x = a.get("x", 500)
        y = a.get("y", 500)
    elif isinstance(a, (list, tuple)) and len(a) >= 2:
        x, y = a[0], a[1]
    else:
        x, y = 500, 500

    try:
        x = float(x)
    except Exception:
        x = 500.0
    try:
        y = float(y)
    except Exception:
        y = 500.0

    x = 0.0 if x < 0 else 1000.0 if x > 1000 else x
    y = 0.0 if y < 0 else 1000.0 if y > 1000 else y
    return x, y

def _parse_text(arg_str: Any) -> str:
    try:
        a = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
    except Exception:
        a = {}
    if isinstance(a, dict):
        t = a.get("text", "")
    else:
        t = ""
    return "" if t is None else str(t)

# ----------------------------
# Input actions (SendInput)
# ----------------------------

def _move_mouse_norm(xn: float, yn: float) -> VirtualScreen:
    vs = _get_virtual_screen()
    x, y = _norm_to_screen_px(xn, yn, vs)
    _dpi_aware()
    ok = user32.SetCursorPos(int(x), int(y))
    if not ok:
        log.warning("SetCursorPos failed winerr=%d", ctypes.get_last_error())
    return vs

def _send_mouse_inputs(inputs: List[INPUT]) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        log.warning("SendInput(mouse) sent=%d expected=%d winerr=%d", sent, len(inputs), ctypes.get_last_error())

def _click_mouse() -> None:
    inp1 = INPUT()
    inp1.type = INPUT_MOUSE
    inp1.ii.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)

    inp2 = INPUT()
    inp2.type = INPUT_MOUSE
    inp2.ii.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)

    _send_mouse_inputs([inp1, inp2])

def _scroll_down_one_notch() -> None:
    wheel_delta = -120
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.ii.mi = MOUSEINPUT(0, 0, ctypes.c_uint32(wheel_delta & 0xFFFFFFFF).value, MOUSEEVENTF_WHEEL, 0, 0)
    _send_mouse_inputs([inp])

def _type_text(text: str) -> None:
    for ch in text:
        code = ord(ch)
        inp = (INPUT * 2)()
        inp[0].type = INPUT_KEYBOARD
        inp[0].ii.ki.wVk = 0
        inp[0].ii.ki.wScan = code
        inp[0].ii.ki.dwFlags = KEYEVENTF_UNICODE
        inp[0].ii.ki.time = 0
        inp[0].ii.ki.dwExtraInfo = 0

        inp[1].type = INPUT_KEYBOARD
        inp[1].ii.ki.wVk = 0
        inp[1].ii.ki.wScan = code
        inp[1].ii.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        inp[1].ii.ki.time = 0
        inp[1].ii.ki.dwExtraInfo = 0

        sent = user32.SendInput(2, inp, ctypes.sizeof(INPUT))
        if sent != 2:
            log.warning("SendInput(keyboard) sent=%d winerr=%d", sent, ctypes.get_last_error())
        time.sleep(0.005)

# ----------------------------
# Conversation helpers
# ----------------------------

def _prune_old_screens(messages: List[Dict[str, Any]], keep_last: int) -> List[Dict[str, Any]]:
    """
    Keep only the last N 'user' multipart messages (screenshots).
    """
    if keep_last <= 0:
        return [m for m in messages if not (m.get("role") == "user" and isinstance(m.get("content"), list))]

    idxs = [i for i, m in enumerate(messages) if m.get("role") == "user" and isinstance(m.get("content"), list)]
    if len(idxs) <= keep_last:
        return messages
    drop = set(idxs[:-keep_last])
    return [m for i, m in enumerate(messages) if i not in drop]

def _disabled_tools_from_schema(tools_schema: List[Dict[str, Any]]) -> set:
    """
    If a tool's description contains 'UNAVAILABLE', treat it as disabled.
    This makes Scenario 8 meaningful even if the schema still includes move_mouse.  (Your test_scenarios does this.)
    """
    disabled = set()
    for t in tools_schema:
        try:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = (fn.get("description") or "")
            if "UNAVAILABLE" in desc.upper():
                disabled.add(name)
        except Exception:
            pass
    return disabled

# ----------------------------
# Defaults (used only if you run without scenario args)
# ----------------------------

SYSTEM_PROMPT = (
    "You control a Windows 11 computer using tool calls only. "
    "Tools: take_screenshot, move_mouse(x,y in 0..1000), click_mouse, type_text(text), scroll_down. "
    "Coordinates are normalized integers 0..1000 relative to the screenshot: (0,0) top-left, (1000,1000) bottom-right. "
    "Screenshots show the mouse cursor position and shape. After move_mouse/click_mouse/type_text/scroll_down, tool responses include "
    "cursor position in pixels and normalized 0..1000. "
    "Workflow: observe (take_screenshot), do ONE action, then observe again."
)

TASK_PROMPT = "Take a screenshot, move mouse to the center of the notepad++ window, click, type hello, take another screenshot and if hello is visible then scroll down once."

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "take_screenshot", "description": "Capture screen and return current view with cursor visible.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "move_mouse", "description": "Move mouse using normalized coordinates 0..1000 relative to the screenshot.", "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {"name": "click_mouse", "description": "Left click at current cursor position.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "type_text", "description": "Type text into the focused control.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "scroll_down", "description": "Scroll down by one notch.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

# ----------------------------
# CLI arg parsing (minimal; keeps your existing interface)
# ----------------------------

def _parse_cli(argv: List[str]) -> Dict[str, Any]:
    """
    Supported forms:
      python script.py
      python script.py test_scenarios.txt 7
    Optional flags (after the two positional args):
      --dump_dir PATH
      --keep_screens N
      --quiet
    """
    cfg: Dict[str, Any] = {}
    if len(argv) >= 3 and not argv[1].startswith("-"):
        cfg["scenario_file"] = argv[1]
        cfg["scenario_num"] = int(argv[2])
        rest = argv[3:]
    else:
        rest = argv[1:]

    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--dump_dir" and i + 1 < len(rest):
            cfg["dump_dir"] = rest[i + 1]
            i += 2
        elif a == "--keep_screens" and i + 1 < len(rest):
            cfg["keep_screens"] = int(rest[i + 1])
            i += 2
        elif a == "--quiet":
            cfg["quiet"] = True
            i += 1
        else:
            log.warning("Unknown arg ignored: %s", a)
            i += 1
    return cfg

# ----------------------------
# Main loop
# ----------------------------

def main() -> None:
    global DUMP_DIR, KEEP_LAST_SCREENSHOTS, VERBOSE_LOG_CONVERSATION

    cli = _parse_cli(sys.argv)

    if cli.get("quiet"):
        VERBOSE_LOG_CONVERSATION = False

    if "dump_dir" in cli:
        DUMP_DIR = cli["dump_dir"]

    if "keep_screens" in cli:
        KEEP_LAST_SCREENSHOTS = max(0, int(cli["keep_screens"]))

    # Defaults
    system_prompt = SYSTEM_PROMPT
    task_prompt = TASK_PROMPT
    tools_schema = TOOLS_SCHEMA

    # Scenario override
    if "scenario_file" in cli and "scenario_num" in cli:
        system_prompt, task_prompt, tools_schema = load_scenario(cli["scenario_file"], cli["scenario_num"])
        log.info("Using scenario %d from %s", cli["scenario_num"], cli["scenario_file"])

    disabled_tools = _disabled_tools_from_schema(tools_schema)
    if disabled_tools:
        log.info("Disabled tools (per schema description): %s", ", ".join(sorted(disabled_tools)))

    os.makedirs(DUMP_DIR, exist_ok=True)

    log.info("Model: %s | Endpoint: %s", MODEL_ID, LM_STUDIO_ENDPOINT)
    log.info("Task prompt: %s", task_prompt)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    dump_idx = int(DUMP_START)
    last_vs = _get_virtual_screen()

    for step in range(MAX_STEPS):
        log.info("=" * 80)
        log.info("STEP %d - Sending request to model", step + 1)

        if VERBOSE_LOG_CONVERSATION:
            log.info("Conversation has %d messages", len(messages))
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append("TEXT: " + part.get("text", ""))
                        elif part.get("type") == "image_url":
                            parts.append("IMAGE: [BASE64_PNG]")
                    log.info("  [%d] %s: %s", i, role.upper(), " | ".join(parts))
                elif isinstance(content, str):
                    display = content[:200] + "..." if len(content) > 200 else content
                    log.info("  [%d] %s: %s", i, role.upper(), display)
                else:
                    log.info("  [%d] %s: [complex content]", i, role.upper())

                if "tool_call_id" in msg:
                    log.info("      └─ tool_call_id: %s", msg["tool_call_id"])

        resp = _post({
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools_schema,
            "tool_choice": "auto",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        })

        msg = resp["choices"][0]["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []

        log.info("STEP %d - Model response: content_len=%d tool_calls=%d",
                 step + 1, len(msg.get("content") or ""), len(tool_calls))

        if msg.get("content"):
            display = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            log.info("  ASSISTANT: %s", display)

        if not tool_calls:
            log.info("Done: No tool calls in response.")
            break

        for tc in tool_calls:
            name = tc["function"]["name"]
            arg_str = tc["function"].get("arguments", "{}")
            call_id = tc["id"]

            if name in disabled_tools:
                log.info("Tool %s is disabled for this scenario; returning error.", name)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": "error tool_disabled"})
                continue

            log.info("Executing tool: %s", name)

            if name == "take_screenshot":
                t0 = time.time()
                png_bytes, vs = take_screenshot_png()
                last_vs = vs
                dt = time.time() - t0

                fn = None
                if DUMP_SCREENSHOTS:
                    fn = os.path.join(DUMP_DIR, f"{DUMP_PREFIX}{dump_idx:04d}.png")
                    with open(fn, "wb") as f:
                        f.write(png_bytes)
                    dump_idx += 1

                cx, cy, cnx, cny = _cursor_pos_norm(vs)
                tool_text = "ok" if fn is None else ("ok file=" + fn)
                log.info("  → Screenshot %dx%d (virtual=%dx%d at %d,%d) in %.3fs saved=%s cursor_px=%d,%d cursor_norm=%d,%d",
                         TARGET_W, TARGET_H, vs.width, vs.height, vs.left, vs.top, dt, fn or "no", cx, cy, cnx, cny)

                b64 = base64.b64encode(png_bytes).decode("ascii")
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": tool_text})

                # Attach image for the model, prune old ones
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "screen"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    ],
                })
                messages = _prune_old_screens(messages, KEEP_LAST_SCREENSHOTS)

            elif name == "move_mouse":
                xn, yn = _parse_norm_xy(arg_str)
                vs = _move_mouse_norm(xn, yn)
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(vs)
                log.info("  → move_mouse norm(%d,%d) cursor_px(%d,%d) cursor_norm(%d,%d)",
                         int(xn), int(yn), cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": "ok cursor_px=%d,%d cursor_norm=%d,%d" % (cx, cy, cnx, cny)})

            elif name == "click_mouse":
                _click_mouse()
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(last_vs)
                log.info("  → click_mouse cursor_px(%d,%d) cursor_norm(%d,%d)", cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": "ok cursor_px=%d,%d cursor_norm=%d,%d" % (cx, cy, cnx, cny)})

            elif name == "type_text":
                text = _parse_text(arg_str)
                _type_text(text)
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(last_vs)
                log.info("  → type_text '%s' cursor_px(%d,%d) cursor_norm(%d,%d)", text, cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": "ok typed=%s cursor_px=%d,%d cursor_norm=%d,%d" % (text, cx, cy, cnx, cny)})

            elif name == "scroll_down":
                _scroll_down_one_notch()
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(last_vs)
                log.info("  → scroll_down cursor_px(%d,%d) cursor_norm(%d,%d)", cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": "ok cursor_px=%d,%d cursor_norm=%d,%d" % (cx, cy, cnx, cny)})

            else:
                log.warning("Unknown tool requested: %s", name)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": "error unknown_tool"})

        time.sleep(STEP_DELAY)

if __name__ == "__main__":
    main()
