import asyncio
import base64
import shlex
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
from anthropic.types.beta import BetaToolComputerUse20241022Param
from screeninfo import get_monitors

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run
import logging

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

MAC_SCALED_RES_HEIGHT = 662
MAC_SCALED_RES_WIDTH = 1024
XGA_RES_HEIGHT = 768
XGA_RES_WIDTH = 1024

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int

##########################
# Xdotool-to-AppleScript Mapping
##########################
XDOTOOL_TO_APPLESCRIPT_KEYCODES = {
    # Movement / Arrows
    "Left": 123,
    "Right": 124,
    "Down": 125,
    "Up": 126,

    # Special keys
    "Return": 36,
    "Enter": 36,
    "Esc": 53,
    "Escape": 53,
    "Tab": 48,
    "BackSpace": 51,  # macOS calls this 'delete'
    "Delete": 117,    # forward delete on mac
    "Home": 115,
    "End": 119,
    "Page_Up": 116,
    "Prior": 116,     # often same as Page_Up
    "Page_Down": 121,
    "Next": 121,      # often same as Page_Down

    # Function keys
    "F1": 122, "F2": 120, "F3": 99,  "F4": 118,
    "F5": 96,  "F6": 97,  "F7": 98,  "F8": 100,
    "F9": 101, "F10": 109, "F11": 103, "F12": 111,
    # (Add more if needed: F13-F16, etc.)
}

XDOTOOL_TO_APPLESCRIPT_MODIFIERS = {
    "ctrl":  "control down",
    "control": "control down",
    "alt":   "option down",
    "shift": "shift down",
    "super": "command down",  # or "cmd" => "command down"
    "cmd":   "command down",
    "meta":  "command down",
}

def press_key_applescript(keycode: int, modifiers: list[str]) -> str:
    """Return an AppleScript snippet to press a given keycode with optional modifiers."""
    # Convert your "cmd"/"ctrl"/"alt"/"shift" to AppleScript equivalents:
    applescript_mods_map = {
        "cmd": "command down",
        "ctrl": "control down",
        "alt": "option down",
        "shift": "shift down",
    }
    # Filter out any modifiers not recognized
    applescript_mods = [applescript_mods_map[m] for m in modifiers if m in applescript_mods_map]
    if applescript_mods:
        mods_str = " using {" + ", ".join(applescript_mods) + "}"
    else:
        mods_str = ""
    
    return f'tell application "System Events" to key code {keycode}{mods_str}'

def press_character_applescript(char: str, modifiers: list[str]) -> str:
    """Return an AppleScript snippet to press a character (e.g. 'a') with optional modifiers."""
    applescript_mods_map = {
        "cmd": "command down",
        "ctrl": "control down",
        "alt": "option down",
        "shift": "shift down",
    }
    applescript_mods = [applescript_mods_map[m] for m in modifiers if m in applescript_mods_map]
    if applescript_mods:
        mods_str = " using {" + ", ".join(applescript_mods) + "}"
    else:
        mods_str = ""
    
    # If char is a double quote, we need to escape it for AppleScript
    safe_char = char.replace('"', '\\"')
    return f'tell application "System Events" to keystroke "{safe_char}"{mods_str}'

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        monitor = get_monitors()[0]
        self.width = monitor.width
        self.height = monitor.height

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                return await self.shell(f"cliclick m:{x},{y}")
            elif action == "left_click_drag":
                await self.shell(f"cliclick dd:.")
                await self.shell(f"cliclick m:{x},{y}")
                return await self.shell(f"cliclick du:.")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                parts = text.split('+')
                # Last item is the "main" key, the rest are modifiers
                key = parts[-1]
                raw_mods = parts[:-1]

                # Convert xdotool-style modifiers to AppleScript strings
                applescript_mods = []
                for m in raw_mods:
                    if m in XDOTOOL_TO_APPLESCRIPT_MODIFIERS:
                        applescript_mods.append(XDOTOOL_TO_APPLESCRIPT_MODIFIERS[m])
                    else:
                        logging.error(f"Unknown modifier '{m}' - ignoring")

                # Decide if this is a "special key" (arrow, Return, etc.):
                if key in XDOTOOL_TO_APPLESCRIPT_KEYCODES:
                    # It's a known special key => use key code
                    keycode = XDOTOOL_TO_APPLESCRIPT_KEYCODES[key]
                    applescript_cmd = press_key_applescript(keycode, applescript_mods)
                else:
                    # Assume it's a normal character (like 'a', 'b', '1', '-', etc.)
                    # We'll press it using keystroke
                    applescript_cmd = press_character_applescript(key, applescript_mods)

                if not applescript_cmd:
                    raise ToolError(f"Cannot execute key '{text}'. Please try a different method.")
                
                return await self.shell(f'osascript -e \'{applescript_cmd}\'')
                                
            elif action == "type":
                result = await self.shell(f'cliclick t:{text}', take_screenshot=False)
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(
                    output=result.output,
                    error=result.error,
                    base64_image=screenshot_base64,
                )

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                result = await self.shell("cliclick p:", take_screenshot=False)
                output = result.output.strip() or ""
                x_int, y_int = map(int, output.split(","))
                x, y = self.scale_coordinates(
                    ScalingSource.COMPUTER,
                    x_int,
                    y_int,
                )
                return result.replace(output=f'X={x},Y={y}')
            else:
                if action == "middle_click":
                    logging.error('Middle click leveraged')
                    return ToolResult(error="Middle click is not supported currently.")
                click_arg = {
                    "left_click": "c:.",
                    "right_click": "rc:.",
                    # "middle_click": "",
                    "double_click": "dc:.",
                }[action]
                return await self.shell(f"cliclick {click_arg}")

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        result = await self.shell(f'screencapture -x {path}', take_screenshot=False)
        await self.shell(
            f'magick {path} \
                -resize {MAC_SCALED_RES_WIDTH}x{MAC_SCALED_RES_HEIGHT} \
                -gravity north \
                -background black \
                -extent {XGA_RES_WIDTH}x{XGA_RES_HEIGHT} \
                {path}', take_screenshot=False
        )

        if path.exists():
            return result.replace(
                base64_image=base64.b64encode(path.read_bytes()).decode()
            )
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        if stderr:
            logging.error('Error from shell command: %s', stderr)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates between actual screen resolution and target scaled resolution.
        Handles conversion between actual Mac resolution (1728x1117), 
        scaled screenshot (1024x662), and final XGA format (1024x768).
        
        When scaling API -> Computer:
            1. Coordinates come in relative to XGA resolution (1024x768)
            2. Need to translate to scaled screenshot coordinates (1024x662)
            3. Then scale up to actual screen coordinates (1728x1117)
        
        When scaling Computer -> API:
            1. Coordinates come in from actual screen (1728x1117)
            2. Scale down to screenshot size (1024x662)
            3. Translate to XGA coordinates (1024x768)
        
        Args:
            source: ScalingSource.COMPUTER for scaling down from real screen coordinates
                ScalingSource.API for scaling up from XGA resolution coordinates
            x: The x coordinate to scale
            y: The y coordinate to scale
            
        Returns:
            tuple[int, int]: The scaled (x, y) coordinates
        """
        screen_to_scaled_x = MAC_SCALED_RES_WIDTH / self.width
        screen_to_scaled_y = MAC_SCALED_RES_HEIGHT / self.height

        if source == ScalingSource.API:
            if x > MAC_SCALED_RES_WIDTH or y > MAC_SCALED_RES_HEIGHT:
                raise ToolError(f'Coordinates {x}, {y} are out of bounds')
                
            actual_x = round(x / screen_to_scaled_x)
            actual_y = round(y / screen_to_scaled_y)
            
            return actual_x, actual_y
        
        else:  # ScalingSource.COMPUTER               
            scaled_x = round(x * screen_to_scaled_x)
            scaled_y = round(y * screen_to_scaled_y)
            
            return scaled_x, scaled_y
