#!/usr/bin/env python3
"""
Convert simplified Chinese transcript to traditional Chinese using OpenCC.
"""

import json
import sys
from pathlib import Path

try:
    from opencc import OpenCC
except ImportError:
    print("Please install opencc: pip install opencc-python-reimplemented")
    sys.exit(1)


def convert_transcript(input_path, output_path=None, config='s2t'):
    """
    Convert transcript from simplified to traditional Chinese.

    Config options:
        s2t  - Simplified to Traditional (default)
        s2tw - Simplified to Traditional (Taiwan standard)
        s2hk - Simplified to Traditional (Hong Kong standard)
        t2s  - Traditional to Simplified
    """

    cc = OpenCC(config)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert segment text and words
    for segment in data.get('segments', []):
        if 'text' in segment:
            segment['text'] = cc.convert(segment['text'])

        if 'words' in segment:
            for word in segment['words']:
                if 'word' in word:
                    word['word'] = cc.convert(word['word'])

    # Convert word_segments if present
    if 'word_segments' in data:
        for word in data['word_segments']:
            if 'word' in word:
                word['word'] = cc.convert(word['word'])

    # Determine output path
    if output_path is None:
        p = Path(input_path)
        suffix = '_traditional' if config.startswith('s2') else '_simplified'
        output_path = p.parent / f"{p.stem}{suffix}{p.suffix}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted transcript saved to: {output_path}")
    return output_path


def preview_conversion(input_path, config='s2t'):
    """Preview conversion without saving."""

    cc = OpenCC(config)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('segments'):
        print("No segments found")
        return

    # Show first segment
    first_segment = data['segments'][0]
    original_text = first_segment.get('text', '')[:100]
    converted_text = cc.convert(original_text)

    print("=" * 60)
    print("ORIGINAL (Simplified):")
    print(original_text)
    print()
    print("CONVERTED (Traditional):")
    print(converted_text)
    print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_to_traditional.py <input.json>              # Convert to Traditional")
        print("  python convert_to_traditional.py <input.json> <output.json>")
        print("  python convert_to_traditional.py --preview <input.json>    # Preview without saving")
        print()
        print("Options (set via --config):")
        print("  s2t  - Simplified → Traditional (default)")
        print("  s2tw - Simplified → Traditional (Taiwan)")
        print("  s2hk - Simplified → Traditional (Hong Kong)")
        print("  t2s  - Traditional → Simplified")
        print()
        print("Example with Taiwan standard:")
        print("  python convert_to_traditional.py --config s2tw input.json")
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    config = 's2t'
    preview = False
    input_file = None
    output_file = None

    i = 0
    while i < len(args):
        if args[i] == '--config' and i + 1 < len(args):
            config = args[i + 1]
            i += 2
        elif args[i] == '--preview':
            preview = True
            i += 1
        elif input_file is None:
            input_file = args[i]
            i += 1
        else:
            output_file = args[i]
            i += 1

    if input_file is None:
        print("Please provide an input file")
        sys.exit(1)

    if preview:
        preview_conversion(input_file, config)
    else:
        convert_transcript(input_file, output_file, config)
