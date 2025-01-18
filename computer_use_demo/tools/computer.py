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

def xdotool_to_cliclick_key_mapping(k: str) -> str:
    # Convert from xdotool naming to cliclick naming
    mapping = {
        "Return": "return",
        "Enter": "return",
        "Esc": "esc",
        "Escape": "esc",
        "Tab": "tab",
        "Up": "arrow-up",
        "Down": "arrow-down",
        "Left": "arrow-left",
        "Right": "arrow-right",
        "BackSpace": "delete",
        "Delete": "fwd-delete",
        "Home": "home",
        "End": "end",
        "Page_Up": "page-up",
        "Prior": "page-up",
        "Page_Down": "page-down",
        "Next": "page-down",
        # Function keys
        "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
        "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
        "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
        "F13": "f13", "F14": "f14", "F15": "f15", "F16": "f16",
        # Numpad
        "KP_0": "num-0", "KP_1": "num-1", "KP_2": "num-2", "KP_3": "num-3",
        "KP_4": "num-4", "KP_5": "num-5", "KP_6": "num-6", "KP_7": "num-7",
        "KP_8": "num-8", "KP_9": "num-9",
        "KP_Add": "num-plus", "KP_Subtract": "num-minus",
        "KP_Multiply": "num-multiply", "KP_Divide": "num-divide",
        "KP_Enter": "num-enter",
        "KP_Equal": "num-equals",
        # Media, brightness, etc. (XF86 keys)
        "XF86MonBrightnessDown": "brightness-down",
        "XF86MonBrightnessUp": "brightness-up",
        "XF86AudioMute": "mute",
        "XF86AudioLowerVolume": "volume-down",
        "XF86AudioRaiseVolume": "volume-up",
        "XF86AudioPlay": "play-pause",
        "XF86AudioNext": "play-next",
        "XF86AudioPrev": "play-previous",
        "XF86KbdBrightnessDown": "keys-light-down",
        "XF86KbdBrightnessUp": "keys-light-up",
    }
    if k in mapping:
        return mapping[k]
    return k.lower()

def xdotool_to_cliclick_modifier_mapping(mod: str) -> str:
    # Convert from xdotool naming to cliclick
    m = mod.lower()
    if m in ["ctrl", "control"]:
        return "ctrl"
    elif m in ["alt", "option"]:
        return "alt"
    elif m in ["super", "cmd", "command"]:
        return "cmd"
    elif m == "shift":
        return "shift"
    return m

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
                logging.info('Key leveraged %s', text)
                parts = text.split('+')
                if len(parts) == 1:
                    # No modifiers, just a single key
                    key = parts[0]
                    return await self.shell(f"cliclick kp:{xdotool_to_cliclick_key_mapping(key)}")
                else:
                    # Everything but the last item is a modifier, the last is the "main" key
                    *mods, main_key = parts
                    mods = [xdotool_to_cliclick_key_mapping(m) for m in mods]
                    main_key = xdotool_to_cliclick_modifier_mapping(main_key)

                    # Press modifiers
                    mods_arg = ",".join(mods)
                    await self.shell(f"cliclick kd:{mods_arg}")
                    # Press the main key
                    await self.shell(f"cliclick kp:{main_key}")
                    # Release modifiers
                    return await self.shell(f"cliclick ku:{mods_arg}")
                                
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
