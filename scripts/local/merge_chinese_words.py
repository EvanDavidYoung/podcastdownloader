#!/usr/bin/env python3
"""
Merge character-level Chinese transcript into word-level using jieba segmentation.
Preserves timestamps by using start of first char and end of last char for each word.
"""

import json
import sys
import re
from pathlib import Path

try:
    import jieba
except ImportError:
    print("Please install jieba: pip install jieba")
    sys.exit(1)


def is_chinese_char(char):
    """Check if character is Chinese."""
    return '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf'


def is_punctuation(char):
    """Check if character is punctuation."""
    return char in '，。！？、；：""''（）【】《》…—·,.!?;:\'"()[]<>'


def merge_words_in_segment(words):
    """
    Merge character-level words into proper Chinese words using jieba.
    Non-Chinese text (English, numbers) is kept as-is.
    """
    if not words:
        return words

    # Build the full text and track character positions
    full_text = ''.join(w['word'] for w in words)

    # Use jieba to segment
    segmented = list(jieba.cut(full_text))

    # Now map segmented words back to original timestamps
    merged_words = []
    char_idx = 0  # Index into original words list

    for seg_word in segmented:
        if not seg_word.strip():
            continue

        seg_len = len(seg_word)

        # Find the range of original words that make up this segmented word
        start_idx = char_idx
        chars_consumed = 0
        end_idx = char_idx

        while chars_consumed < seg_len and end_idx < len(words):
            chars_consumed += len(words[end_idx]['word'])
            end_idx += 1

        if start_idx < len(words) and end_idx <= len(words):
            # Merge timestamps: start from first, end from last
            merged_word = {
                'word': seg_word,
                'start': words[start_idx]['start'],
                'end': words[end_idx - 1]['end'],
                'score': sum(w.get('score', 1.0) for w in words[start_idx:end_idx]) / (end_idx - start_idx)
            }
            merged_words.append(merged_word)

        char_idx = end_idx

    return merged_words


def process_transcript(input_path, output_path=None):
    """Process a transcript JSON file."""

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each segment
    for segment in data.get('segments', []):
        if 'words' in segment:
            segment['words'] = merge_words_in_segment(segment['words'])

    # Also update word_segments if present
    if 'word_segments' in data:
        data['word_segments'] = merge_words_in_segment(data['word_segments'])

    # Determine output path
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_merged{p.suffix}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Merged transcript saved to: {output_path}")
    return output_path


def preview_merge(input_path, num_words=50):
    """Preview the merge without saving."""

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('segments'):
        print("No segments found")
        return

    # Get first segment's words
    first_segment = data['segments'][0]
    original_words = first_segment.get('words', [])[:num_words]
    merged_words = merge_words_in_segment(original_words)

    print("=" * 60)
    print("ORIGINAL (character-level):")
    print(" | ".join(w['word'] for w in original_words[:20]))
    print()
    print("MERGED (word-level):")
    print(" | ".join(w['word'] for w in merged_words[:20]))
    print("=" * 60)
    print(f"\nOriginal: {len(original_words)} tokens → Merged: {len(merged_words)} tokens")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python merge_chinese_words.py <input.json>           # Save to input_merged.json")
        print("  python merge_chinese_words.py <input.json> <output.json>  # Save to specified path")
        print("  python merge_chinese_words.py --preview <input.json>      # Preview without saving")
        sys.exit(1)

    if sys.argv[1] == '--preview':
        if len(sys.argv) < 3:
            print("Please provide input file for preview")
            sys.exit(1)
        preview_merge(sys.argv[2])
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        process_transcript(input_file, output_file)
