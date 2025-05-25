import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
from collections import defaultdict
from typing import List, Tuple, Dict

# 4 thuật toán khớp chuỗi cơ bản
def brute_force(text, pattern):
    n, m = len(text), len(pattern)
    result, comparisons = [], 0
    for i in range(n - m + 1):
        comparisons += 1
        match = True
        for j in range(m):
            comparisons += 1
            comparisons += 1
            if text[i + j] != pattern[j]:
                match = False
                break
        comparisons += 1
        if match:
            result.append(i)
    return result, comparisons


def boyer_moore(text, pattern):
    def bad_char_table(pat):
        table = {}
        for i in range(len(pat)):
            table[pat[i]] = i
        return table

    n, m = len(text), len(pattern)
    table = bad_char_table(pattern)
    result, comparisons = [], 0

    s = 0
    while s <= n - m:
        comparisons += 1
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            comparisons += 1
            j -= 1
        comparisons+=1
        if j < 0:
            result.append(s)
            s += m - table.get(text[s + m], -1) if s + m < n else 1
        else:
            s += max(1, j - table.get(text[s + j], -1))
    return result, comparisons

class OptimizedRabinKarp:    
    def __init__(self, base: int = 256, prime: int = 1000000007):
        self.base = base
        self.prime = prime
        self.hash_cache = {}
        self.bloom_filter_size = 1024
        
    def polynomial_hash(self, text: str, length: int) -> int:
        hash_val = 0
        for i, char in enumerate(text[:length]):
            hash_val = (hash_val * self.base + ord(char)) % self.prime
        return hash_val
    
    def rolling_hash_update(self, old_hash: int, old_char: str, new_char: str, power: int) -> int:
        old_hash = (old_hash - ord(old_char) * power) % self.prime
        old_hash = (old_hash * self.base + ord(new_char)) % self.prime
        return (old_hash + self.prime) % self.prime
    
    def _bloom_hash1(self, text: str) -> int:
        return sum(ord(c) for c in text) % self.bloom_filter_size
    
    def _bloom_hash2(self, text: str) -> int:
        return sum(i * ord(c) for i, c in enumerate(text)) % self.bloom_filter_size
    
    def single_pattern_search(self, text: str, pattern: str) -> Tuple[List[int], int]:
        n, m = len(text), len(pattern)
        if m > n or m == 0:
            return [], 0
        result = []
        comparisons = 0
        window_hashes = {}
        power = pow(self.base, m - 1, self.prime)
        pattern_hash = self.polynomial_hash(pattern, m)
        text_hash = self.polynomial_hash(text, m)
        window_hashes[0] = text_hash
        if pattern_hash == text_hash:
            comparisons += 1
            if text[:m] == pattern:
                result.append(0)
        for i in range(1, n - m + 1):
            comparisons += 1
            if i in window_hashes:
                text_hash = window_hashes[i]
            else:
                text_hash = self.rolling_hash_update(
                    text_hash, text[i-1], text[i+m-1], power
                )
                window_hashes[i] = text_hash
            comparisons+=1
            if pattern_hash == text_hash:
                comparisons += 1
                if self._fast_compare(text, pattern, i):
                    result.append(i)
        
        return result, comparisons
    
    def search_with_bloom_filter(self, text: str, pattern: str) -> Tuple[List[int], int]:
        """Tìm kiếm với Bloom filter pre-filtering"""
        n, m = len(text), len(pattern)
        if m > n or m == 0:
            return [], 0
            
        result = []
        comparisons = 0
        
        # Create Bloom filter signature for pattern
        pattern_bloom = {self._bloom_hash1(pattern), self._bloom_hash2(pattern)}
        
        # Rolling hash setup
        power = pow(self.base, m - 1, self.prime)
        pattern_hash = self.polynomial_hash(pattern, m)
        text_hash = self.polynomial_hash(text, m)
        
        # Check first window
        window = text[:m]
        window_bloom = {self._bloom_hash1(window), self._bloom_hash2(window)}
        comparisons += 1
        if pattern_bloom.issubset(window_bloom):
            comparisons+=1
            if pattern_hash == text_hash:
                comparisons += 1
                if window == pattern:
                    result.append(0)
        
        # Slide through text
        for i in range(1, n - m + 1):
            window = text[i:i+m]
            window_bloom = {self._bloom_hash1(window), self._bloom_hash2(window)}
            
            # Bloom filter check first
            comparisons += 1
            if pattern_bloom.issubset(window_bloom):
                text_hash = self.rolling_hash_update(
                    text_hash, text[i-1], text[i+m-1], power
                )
                comparisons += 1
                if pattern_hash == text_hash:
                    comparisons += 1
                    if self._fast_compare(text, pattern, i):
                        result.append(i)
            else:
                # Still need to update hash for next iteration
                text_hash = self.rolling_hash_update(
                    text_hash, text[i-1], text[i+m-1], power
                )
        
        return result, comparisons
    
    def multiple_pattern_search(self, text: str, patterns: List[str]) -> Tuple[Dict[str, List[int]], int]:
        """Tìm kiếm nhiều patterns cùng lúc"""
        if not patterns:
            return {}, 0
            
        n = len(text)
        result = {pattern: [] for pattern in patterns}
        total_comparisons = 0
        
        # Group patterns by length for efficiency
        patterns_by_length = defaultdict(list)
        for pattern in patterns:
            total_comparisons += 1
            if pattern:  # Skip empty patterns
                patterns_by_length[len(pattern)].append(pattern)
        
        # Process each length group
        for pattern_length, pattern_group in patterns_by_length.items():
            total_comparisons += 1
            if pattern_length > n:
                continue
                
            # Compute pattern hashes
            pattern_hashes = {}
            for pattern in pattern_group:
                pattern_hashes[self.polynomial_hash(pattern, pattern_length)] = pattern
            
            # Precompute power
            power = pow(self.base, pattern_length - 1, self.prime)
            
            # Compute first window hash
            total_comparisons += 1
            if pattern_length <= n:
                text_hash = self.polynomial_hash(text, pattern_length)
                
                # Check first window
                total_comparisons += 1
                if text_hash in pattern_hashes:
                    pattern = pattern_hashes[text_hash]
                    total_comparisons += 1
                    if text[:pattern_length] == pattern:
                        result[pattern].append(0)
                
                # Slide through text
                for i in range(1, n - pattern_length + 1):
                    text_hash = self.rolling_hash_update(
                        text_hash, text[i-1], text[i+pattern_length-1], power
                    )
                    total_comparisons += 1
                    if text_hash in pattern_hashes:
                        pattern = pattern_hashes[text_hash]
                        total_comparisons += 1
                        if self._fast_compare(text, pattern, i):
                            result[pattern].append(i)
        
        return result, total_comparisons
    
    def _fast_compare(self, text: str, pattern: str, start: int) -> bool:
        """So sánh ký tự nhanh với early exit"""
        for i in range(len(pattern)):
            if text[start + i] != pattern[i]:
                return False
        return True

def rabin_karp_original(text, pattern, d=20, q=101):
    n, m = len(text), len(pattern)
    if m > n:
        return [], 0
    result, comparisons = [], 0
    h = pow(d, m - 1, q)
    p = sum((d ** (m - 1 - i)) * ord(pattern[i]) for i in range(m)) % q
    t = sum((d ** (m - 1 - i)) * ord(text[i]) for i in range(m)) % q

    for s in range(n - m + 1):
        comparisons += 1
        comparisons += 1
        if p == t:
            match = True
            for i in range(m):
                comparisons += 1
                comparisons += 1
                if text[s + i] != pattern[i]:
                    match = False
                    break
            comparisons += 1
            if match:
                
                result.append(s)
        comparisons += 1
        if s < n - m:
            t = (t - ord(text[s]) * h) % q
            t = (t * d + ord(text[s + m])) % q
            t = (t + q) % q
    return result, comparisons

class Nadpote:
    def __init__(self, root):
        self.root = root
        self.root.title("Nadpote--")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.root.iconbitmap("logo.ico")

        self.rk_optimizer = OptimizedRabinKarp()
        
        self.create_widgets()
        self.match_positions = []
        self.current_match_index = -1
        self.multi_pattern_results = {}



    def create_widgets(self):
        # Thanh menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu.add_cascade(label="File", menu=file_menu)
        
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="Pattern:").pack(side=tk.LEFT)
        self.pattern_entry = tk.Entry(top_frame, width=30)
        self.pattern_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_choice = ttk.Combobox(top_frame, values=[
            "Brute Force", 
            "Rabin-Karp", 
            "Boyer-Moore", 
        ])
        self.algo_choice.current(2) 
        self.algo_choice.pack(side=tk.LEFT, padx=5)
        from PIL import Image, ImageTk, ImageEnhance

        def enhance_image(image, factor):
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)

        def add_hover_effect(button, original_image):
            # Tạo ảnh sẫm màu
            dark_image = enhance_image(original_image, 0.6)
            dark_icon = ImageTk.PhotoImage(dark_image)

            def on_enter(event):
                button.config(image=dark_icon)
                button.image_hover = dark_icon  # giữ tham chiếu

            def on_leave(event):
                button.config(image=button.image_original)

            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)


        img_search = Image.open("search.png").resize((20, 20))
        icon_search = ImageTk.PhotoImage(img_search)
        search_button = tk.Button(top_frame, image=icon_search, command=self.search_pattern)
        search_button.pack(side=tk.LEFT, padx=5)
        search_button.image_original = icon_search
        add_hover_effect(search_button, img_search)

        # Button Next
        img_next = Image.open("next.webp").resize((20, 20))
        icon_next = ImageTk.PhotoImage(img_next)
        next_button = tk.Button(top_frame, image=icon_next, command=self.find_next)
        next_button.pack(side=tk.LEFT, padx=2)
        next_button.image_original = icon_next
        add_hover_effect(next_button, img_next)

        # Button Previous
        img_prev = Image.open("pre.jpg").resize((20, 20))
        icon_prev = ImageTk.PhotoImage(img_prev)
        prev_button = tk.Button(top_frame, image=icon_prev, command = self.find_previous)
        prev_button.pack(side=tk.LEFT, padx=2)
        prev_button.image_original = icon_prev
        add_hover_effect(prev_button, img_prev)

        


        # Thanh tìm kiếm đa pattern
        multi_frame = tk.Frame(self.root)
        multi_frame.pack(fill=tk.X, padx=10, pady=2)
        
        tk.Label(multi_frame, text="Multiple Patterns (comma-separated):").pack(side=tk.LEFT)
        self.multi_pattern_entry = tk.Entry(multi_frame, width=50)
        self.multi_pattern_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(multi_frame, text="Multi-Search", command=self.search_multiple_patterns).pack(side=tk.LEFT, padx=5)
        tk.Button(multi_frame, text="Clear All", command=self.clear_highlights).pack(side=tk.LEFT, padx=5)

        # Notebook cho tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab chính - Text editor
        main_frame = tk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Text Editor")

        # Text widget với scrollbar
        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(text_frame, wrap=tk.WORD, undo=True, font=("Consolas", 11))
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text.yview)

        # Tab kết quả
        results_frame = tk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Search Results")

        # Results text widget
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        results_scrollbar = tk.Scrollbar(self.results_text)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.results_text.yview)

        # Tab so sánh performance
        perf_frame = tk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance Comparison")

        self.perf_text = tk.Text(perf_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.perf_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        perf_scrollbar = tk.Scrollbar(self.perf_text)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.perf_text.config(yscrollcommand=perf_scrollbar.set)
        perf_scrollbar.config(command=self.perf_text.yview)

        tk.Button(perf_frame, text="Run Performance Test", 
                 command=self.run_performance_test).pack(pady=5)

        # Khung kết quả
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        self.result_label = tk.Label(bottom_frame, text="Ready", anchor="w")
        self.result_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("All text files", "*.txt *.md *.py *.csv *.log"),
            ("Text files", "*.txt"),
            ("Markdown files", "*.md"),
            ("Python files", "*.py"),
            ("CSV files", "*.csv"),
            ("Log files", "*.log"),
            ("All files", "*.*")
        ])

        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text.delete("1.0", tk.END)
                self.text.insert(tk.END, content)
                self.result_label.config(text=f"Loaded: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")

    def save_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[
                ("All text files", "*.txt *.md *.py *.csv *.log"),
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("Python files", "*.py"),
                ("CSV files", "*.csv"),
                ("Log files", "*.log"),
                ("All files", "*.*")
            ])

        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.text.get("1.0", tk.END))
                self.result_label.config(text=f"Saved: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")

    def search_pattern(self):
        pattern = self.pattern_entry.get()
        text = self.text.get("1.0", tk.END)
        algo = self.algo_choice.get()

        if not pattern:
            messagebox.showwarning("Warning", "Please enter a pattern to search.")
            return

        self.clear_highlights()

        start_time = time.time()

        if algo == "Brute Force":
            matches, comparisons = brute_force(text, pattern)
        elif algo == "Rabin-Karp":
            matches, comparisons = rabin_karp_original(text, pattern)
        elif algo == "Rabin-Karp (Optimized)":
            matches, comparisons = self.rk_optimizer.single_pattern_search(text, pattern)
        elif algo == "Rabin-Karp (Bloom Filter)":
            matches, comparisons = self.rk_optimizer.search_with_bloom_filter(text, pattern)
        elif algo == "Boyer-Moore":
            matches, comparisons = boyer_moore(text, pattern)
        elif algo == "KMP":
            matches, comparisons = kmp(text, pattern)
        else:
            matches, comparisons = [], 0

        end_time = time.time()
        elapsed = round((end_time - start_time) * 1000, 3)

        # Highlight matches
        for pos in matches:
            start = f"1.0+{pos}c"
            end = f"1.0+{pos+len(pattern)}c"
            self.text.tag_add("highlight", start, end)

        self.text.tag_config("highlight", background="yellow", foreground="black")

        self.result_label.config(
            text=f"Algorithm: {algo} | Matches: {len(matches)} | Time: {elapsed} ms | Comparisons: {comparisons:,}"
        )
        
        # Update results tab
        self.update_results_tab(algo, pattern, matches, elapsed, comparisons)
        
        self.match_positions = matches
        self.current_match_index = 0 if matches else -1

    def search_multiple_patterns(self):
        patterns_str = self.multi_pattern_entry.get()
        if not patterns_str:
            messagebox.showwarning("Warning", "Please enter patterns to search.")
            return

        patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
        text = self.text.get("1.0", tk.END)

        self.clear_highlights()

        start_time = time.time()
        results, comparisons = self.rk_optimizer.multiple_pattern_search(text, patterns)
        end_time = time.time()
        elapsed = round((end_time - start_time) * 1000, 3)

        colors = ["yellow", "orange", "pink", "lightblue", "lightgreen", "lightgray", "orange", "lightgray"]
        color_index = 0
        
        total_matches = 0
        for pattern, matches in results.items():
            color = colors[color_index % len(colors)]
            for pos in matches:
                start = f"1.0+{pos}c"
                end = f"1.0+{pos+len(pattern)}c"
                self.text.tag_add(f"highlight_{pattern}", start, end)
            
            self.text.tag_config(f"highlight_{pattern}", background=color, foreground="black")
            total_matches += len(matches)
            color_index += 1

        self.result_label.config(
            text=f"Multi-Pattern Search | Total Matches: {total_matches} | Time: {elapsed} ms | Comparisons: {comparisons:,}"
        )

        self.update_multi_results_tab(patterns, results, elapsed, comparisons)
        self.multi_pattern_results = results

    def update_results_tab(self, algo, pattern, matches, elapsed, comparisons):
        self.results_text.delete("1.0", tk.END)
        
        results = f"Search Results\n{'='*50}\n"
        results += f"Algorithm: {algo}\n"
        results += f"Pattern: '{pattern}'\n"
        results += f"Matches Found: {len(matches)}\n"
        results += f"Execution Time: {elapsed} ms\n"
        results += f"Comparisons: {comparisons:,}\n"
        results += f"Text Length: {len(self.text.get('1.0', tk.END)):,} characters\n\n"
        
        if matches:
            results += "Match Positions:\n"
            results += "-" * 20 + "\n"
            for i, pos in enumerate(matches[:100]):  # Limit to first 100 matches
                line_start = self.text.get("1.0", f"1.0+{pos}c").count('\n') + 1
                results += f"{i+1:3d}. Position {pos:6d} (Line {line_start})\n"
            
            if len(matches) > 100:
                results += f"... and {len(matches) - 100} more matches\n"
        
        self.results_text.insert("1.0", results)

    def update_multi_results_tab(self, patterns, results, elapsed, comparisons):
        self.results_text.delete("1.0", tk.END)
        
        results_text = f"Multi-Pattern Search Results\n{'='*50}\n"
        results_text += f"Patterns: {', '.join(patterns)}\n"
        results_text += f"Execution Time: {elapsed} ms\n"
        results_text += f"Total Comparisons: {comparisons:,}\n\n"
        
        for pattern, matches in results.items():
            results_text += f"Pattern '{pattern}': {len(matches)} matches\n"
            if matches:
                results_text += f"  Positions: {matches[:10]}"  # First 10 positions
                if len(matches) > 10:
                    results_text += f" ... (+{len(matches)-10} more)"
                results_text += "\n"
            results_text += "\n"
        
        self.results_text.insert("1.0", results_text)

    # def run_performance_test(self):
    #     """Chạy test so sánh performance giữa các thuật toán"""
    #     text = self.text.get("1.0", tk.END)
    #     if not text.strip():
    #         messagebox.showwarning("Warning", "Please load some text first.")
    #         return

    #     pattern = self.pattern_entry.get() or "the"  # Default pattern
        
    #     self.perf_text.delete("1.0", tk.END)
    #     self.perf_text.insert("1.0", "Running performance tests...\n")
    #     self.root.update()

    #     algorithms = [
    #         ("Brute Force", brute_force),
    #         ("Rabin-Karp", rabin_karp_original),
    #         ("Boyer-Moore", boyer_moore),
    #     ]

    #     results = []
    #     for name, func in algorithms:
    #         start_time = time.time()
    #         matches, comparisons = func(text, pattern)
    #         end_time = time.time()
    #         elapsed = (end_time - start_time) * 1000
    #         results.append((name, len(matches), elapsed, comparisons))

    #     # Display results
    #     self.perf_text.delete("1.0", tk.END)
        
    #     perf_report = f"Performance Comparison\n{'='*60}\n"
    #     perf_report += f"Text Length: {len(text):,} characters\n"
    #     perf_report += f"Pattern: '{pattern}' (Length: {len(pattern)})\n\n"
        
    #     perf_report += f"{'Algorithm':<25} {'Matches':<8} {'Time (ms)':<12} {'Comparisons':<15}\n"
    #     perf_report += "-" * 60 + "\n"
        
    #     for name, matches, elapsed, comparisons in results:
    #         perf_report += f"{name:<25} {matches:<8} {elapsed:<12.3f} {comparisons:<15,}\n"
        
    #     # Find best performance
    #     fastest = min(results, key=lambda x: x[2])
    #     least_comparisons = min(results, key=lambda x: x[3])
        
    #     perf_report += "\n" + "="*60 + "\n"
    #     perf_report += f"Fastest Algorithm: {fastest[0]} ({fastest[2]:.3f} ms)\n"
    #     perf_report += f"Fewest Comparisons: {least_comparisons[0]} ({least_comparisons[3]:,} comparisons)\n"
        
        
        
    #     self.perf_text.insert("1.0", perf_report)
    
    def run_performance_test(self):
        """Chạy test so sánh performance giữa các thuật toán trên các bộ dữ liệu trong thư mục datasets"""
        datasets_dir = "datasets"
        import os
        if not os.path.exists(datasets_dir):
            messagebox.showerror("Error", f"Directory '{datasets_dir}' not found.")
            return

        self.perf_text.delete("1.0", tk.END)
        self.perf_text.insert("1.0", "Running performance tests on datasets...\n")
        self.root.update()

        # Định nghĩa các giá trị cho các tham số
        text_lengths = [10000, 10000000]
        pattern_lengths = [5, 200]
        alphabets = [4, 62, 256]
        repeats = [1, 50, 80]

        algorithms = [
            ("Brute Force", brute_force),
            ("Rabin-Karp", rabin_karp_original),
            ("Boyer-Moore", boyer_moore),
        ]

        results = []
        for text_len in text_lengths:
            for pat_len in pattern_lengths:
                for alpha in alphabets:
                    for rep in repeats:
                        dir_name = f"TextLen-{text_len}_PatternLen-{pat_len}_Alphabet-{alpha}_Repeat-{rep}pct"
                        dir_path = os.path.join(datasets_dir, dir_name)
                        
                        if not os.path.exists(dir_path):
                            continue

                        text_file = os.path.join(dir_path, "text.txt")
                        pattern_file = os.path.join(dir_path, "pattern.txt")

                        if not (os.path.exists(text_file) and os.path.exists(pattern_file)):
                            continue

                        try:
                            with open(text_file, "r", encoding="utf-8") as f:
                                text = f.read()
                            with open(pattern_file, "r", encoding="utf-8") as f:
                                pattern = f.read().rstrip()  # Xóa khoảng trắng cuối
                        except Exception as e:
                            self.perf_text.insert(tk.END, f"Error reading files in {dir_name}: {e}\n")
                            continue

                        self.perf_text.insert(tk.END, f"\nTesting {dir_name}\n{'-'*60}\n")
                        self.perf_text.insert(tk.END, f"Text Length: {len(text):,}\n")
                        self.perf_text.insert(tk.END, f"Pattern Length: {len(pattern)}\n")
                        self.root.update()

                        dir_results = []
                        for name, func in algorithms:
                            start_time = time.time()
                            matches, comparisons = func(text, pattern)
                            end_time = time.time()
                            elapsed = (end_time - start_time) * 1000
                            dir_results.append((name, len(matches), elapsed, comparisons))
                            self.perf_text.insert(tk.END, f"{name}: {len(matches)} matches, {elapsed:.3f} ms, {comparisons:,} comparisons\n")
                            self.root.update()

                        results.append((dir_name, dir_results))

        # Hiển thị tóm tắt
        self.perf_text.insert(tk.END, "\n" + "="*60 + "\nSummary of All Tests\n" + "="*60 + "\n")
        for dir_name, dir_results in results:
            self.perf_text.insert(tk.END, f"\nDataset: {dir_name}\n{'-'*60}\n")
            self.perf_text.insert(tk.END, f"{'Algorithm':<25} {'Matches':<8} {'Time (ms)':<12} {'Comparisons':<15}\n")
            self.perf_text.insert(tk.END, "-" * 60 + "\n")
            
            for name, matches, elapsed, comparisons in dir_results:
                self.perf_text.insert(tk.END, f"{name:<25} {matches:<8} {elapsed:<12.3f} {comparisons:<15,}\n")
            
            fastest = min(dir_results, key=lambda x: x[2])
            least_comparisons = min(dir_results, key=lambda x: x[3])
            self.perf_text.insert(tk.END, f"\nFastest Algorithm: {fastest[0]} ({fastest[2]:.3f} ms)\n")
            self.perf_text.insert(tk.END, f"Fewest Comparisons: {least_comparisons[0]} ({least_comparisons[3]:,} comparisons)\n")

        self.perf_text.see(tk.END)

    def clear_highlights(self):
        """Xóa tất cả highlights"""
        for tag in self.text.tag_names():
            if "highlight" in tag:
                self.text.tag_remove(tag, "1.0", tk.END)

    def find_next(self):
        if not self.match_positions:
            return
        self.current_match_index = (self.current_match_index + 1) % len(self.match_positions)
        self.scroll_to_match()

    def find_previous(self):
        if not self.match_positions:
            return
        self.current_match_index = (self.current_match_index - 1) % len(self.match_positions)
        self.scroll_to_match()

    def scroll_to_match(self):
        if not self.match_positions or self.current_match_index == -1:
            return
            
        pos = self.match_positions[self.current_match_index]
        self.text.see(f"1.0+{pos}c")  # Cuộn đến vị trí match
        self.text.focus_set()  # Đặt focus vào text widget

if __name__ == "__main__":
    root = tk.Tk()
    app = Nadpote(root)
    root.mainloop()