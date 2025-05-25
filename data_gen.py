import random
import string
import os
from collections import Counter
import math

# Các tham số
length_texts = [10_000, 10_000_000]
length_patterns = [5, 200]
alphabet_sizes = [4, 62]
repeat_ratios = [0.01, 0.8]

def repeat_ratio_to_str(r):
    return f"{int(r*100)}pct"

# Tạo bảng chữ cái
def create_alphabet(size):
    base = string.ascii_letters + string.digits + string.punctuation + " "
    if size <= len(base):
        return list(base[:size])
    extended = base + ''.join(chr(i) for i in range(256, 256 + size))
    return list(extended[:size])

# Sinh từ ngẫu nhiên
def generate_word(alphabet, min_len=3, max_len=12):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(alphabet, k=length))

# Sinh văn bản theo tỷ lệ lặp chính xác
def generate_text_v2(alphabet, total_length, repeat_ratio):
    avg_word_len = 7 
    total_words = math.ceil(total_length / avg_word_len)

    num_repeat_words = int(total_words * repeat_ratio)
    num_new_words = total_words - num_repeat_words

    new_words = [generate_word(alphabet) for _ in range(num_new_words)]
    if new_words:
        repeat_words = [random.choice(new_words) for _ in range(num_repeat_words)]
    else:
        repeat_words = []

    all_words = new_words + repeat_words
    random.shuffle(all_words)

    text = ""
    idx = 0
    while len(text) < total_length and idx < len(all_words):
        text += ' ' + all_words[idx]
        idx += 1
    text = text[:total_length]

    freq = Counter(all_words[:idx])
    vocab = list(set(all_words[:idx]))

    return text, vocab, freq

def generate_pattern_from_text(text, pattern_length):
    if pattern_length > len(text):
        pattern_length = len(text)
    start = random.randint(0, len(text) - pattern_length)
    return text[start:start + pattern_length]

root_dir = "datasets"
os.makedirs(root_dir, exist_ok=True)

for len_text in length_texts:
    for len_pat in length_patterns:
        for alpha_size in alphabet_sizes:
            for repeat_ratio in repeat_ratios:
                rep_str = repeat_ratio_to_str(repeat_ratio)
                folder_name = f"TextLen-{len_text}_PatternLen-{len_pat}_Alphabet-{alpha_size}_Repeat-{rep_str}"
                folder_path = os.path.join(root_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                print(f"Generating dataset in {folder_path} ...")

                alphabet = create_alphabet(alpha_size)
                text, vocab, freq = generate_text_v2(alphabet, len_text, repeat_ratio)
                pattern = generate_pattern_from_text(text, len_pat)

                with open(os.path.join(folder_path, "text.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
                with open(os.path.join(folder_path, "pattern.txt"), "w", encoding="utf-8") as f:
                    f.write(pattern)
                with open(os.path.join(folder_path, "vocab.txt"), "w", encoding="utf-8") as f:
                    for w in vocab:
                        f.write(w + "\n")
                with open(os.path.join(folder_path, "freq.txt"), "w", encoding="utf-8") as f:
                    for w, c in freq.items():
                        f.write(f"{w} {c}\n")

print("Finished generating all datasets.")
