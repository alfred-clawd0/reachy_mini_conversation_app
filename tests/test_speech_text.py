"""Tests for text prepared for local speech synthesis."""

import pytest

from reachy_mini_conversation_app.speech_text import normalize_for_speech


def test_normalize_for_speech_handles_weather_notation() -> None:
    """Verify normalize for speech handles weather notation."""
    text = "There is a 20–35% chance from 5–10 p.m., around 72°F."
    assert normalize_for_speech(text) == (
        "There is a 20 to 35 percent chance from 5 to 10 p.m., around 72 degrees Fahrenheit."
    )


def test_normalize_for_speech_removes_display_only_syntax() -> None:
    """Verify normalize for speech removes display only syntax."""
    text = "## Result\n- Open [Bear](https://bear.app/) at `/Applications/Bear.app/Contents/MacOS`. (×3)"
    assert normalize_for_speech(text) == "Result Open Bear at the application."


def test_normalize_for_speech_preserves_plain_conversation() -> None:
    """Verify normalize for speech preserves plain conversation."""
    text = "That's a good question. I'd bring a light jacket."
    assert normalize_for_speech(text) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-5°C tonight.", "minus 5 degrees Celsius tonight."),
        ("-3% on the day.", "minus 3 percent on the day."),
        ("−5°C tonight.", "minus 5 degrees Celsius tonight."),
        ("- Intro -", "Intro"),
        ("Visit www.example.com.", "Visit the link."),
        ("Visit https://example.com/search?q=weather&units=c.", "Visit the link."),
        ("(https://example.com/path).", "(the link)."),
        ("Open /Users/me/notes.txt.", "Open notes.txt."),
        ("Open /tmp/report.csv.", "Open report.csv."),
        ("Open /private/var/report.csv.", "Open report.csv."),
        ("Open /Users/me/Tools/Editor.app/Contents/MacOS.", "Open the application."),
        ("5 * 3 = 15.", "5 * 3 = 15."),
        ("This is **bold** and *emphasized*.", "This is bold and emphasized."),
        ("Call 555—1234.", "Call 555—1234."),
    ],
)
def test_normalization_preserves_meaning(text, expected):
    """Preserve numerical signs, arithmetic, punctuation, and the kind of path."""
    assert normalize_for_speech(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("10−5=5", "10 minus 5=5"),
        ("It is -5 today.", "It is minus 5 today."),
        ("5-10 on 2026-09-11, score 3-2.", "5-10 on 2026-09-11, score 3-2."),
        ("Open /Applications/Google Chrome.app.", "Open the application."),
        ("Open /Applications/Google Chrome.app/Contents/MacOS.", "Open the application."),
        ("Saved to /tmp/output/", "Saved to output"),
        ("Open /tmp/this_filename_is_much_too_long_to_pronounce.json.", "Open a file."),
        ("1. First\n2. Second", "First, Second"),
        ("**5 items** left", "5 items left"),
        ("It is **72°F** today", "It is 72 degrees Fahrenheit today"),
        ("**Note:** be careful", "Note: be careful"),
        ("**Warning!** Hot.", "Warning! Hot."),
        ("*Step 1:* mix", "Step 1: mix"),
        ("**$5**", "$5"),
        ("_Done._", "Done."),
        ("5 * 3 = 15.", "5 * 3 = 15."),
        ("2*3*4", "2*3*4"),
        ("a * b * c", "a * b * c"),
        ("snake_case_name", "snake_case_name"),
        ("Open /Applications/Visual Studio Code.app.", "Open the application."),
        (
            "Move it to /Applications/ and then launch Xcode.app.",
            "Move it to /Applications/ and then launch Xcode.app.",
        ),
        (
            "Install it into /Applications/Utilities, then open Terminal.app to finish.",
            "Install it into Utilities, then open Terminal.app to finish.",
        ),
        ("***bold italic***", "bold italic"),
        ("Price*", "Price*"),
        ("__init__", "__init__"),
        ("__important words__", "important words"),
    ],
)
def test_speech_wording_and_symbols(text, expected):
    """Keep identifiers and numeric separators while making prose and paths speakable."""
    assert normalize_for_speech(text) == expected
