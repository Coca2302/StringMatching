import random
import string

random.seed(42)

# Tổng độ dài mục tiêu của pathname
TARGET_PATH_LENGTH = 120

# Tỉ lệ phần prefix giống nhau
PREFIX_RATIO = 0.75
PREFIX_LENGTH = int(TARGET_PATH_LENGTH * PREFIX_RATIO)

# Sinh prefix cố định có độ dài ~90 ký tự
def generate_fixed_prefix():
    # Tạo các thư mục con để ghép thành prefix
    prefix_parts = []
    current_len = 0
    while current_len < PREFIX_LENGTH:
        part = random_dirname(random.randint(4, 8))
        prefix_parts.append(part)
        current_len += len(part) + 1  # +1 cho dấu /
    return "/" + "/".join(prefix_parts)

# Sinh một tên thư mục ngẫu nhiên
def random_dirname(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Sinh phần suffix sao cho tổng chiều dài pathname xấp xỉ mục tiêu
def generate_suffix(prefix_len, max_len):
    remaining_len = max_len - prefix_len - 5  # trừ đi .txt
    suffix = []
    current_len = 0
    while current_len < remaining_len:
        part_len = random.randint(3, 6)
        part = random_dirname(part_len)
        suffix.append(part)
        current_len += part_len + 1
    filename = random_dirname(4) + ".txt"
    return "/".join(suffix) + "/" + filename

# Sinh danh sách pathnames
def generate_pathnames(n=100000):
    pathnames = []
    prefix = generate_fixed_prefix()
    for _ in range(n):
        suffix = generate_suffix(len(prefix), TARGET_PATH_LENGTH)
        path = f"{prefix}/{suffix}"
        pathnames.append(path)
    return pathnames

# Sinh dữ liệu
pathnames = generate_pathnames()

# Trích một pattern từ phần prefix giống nhau
example_path = pathnames[0]
pattern = example_path[5:25]

print("Ví dụ path:", example_path)
print("Pattern trích từ phần giống nhau:", pattern)
print(f"Tỷ lệ prefix ~ {len(example_path.split('/')[1:-1][0]) / len(example_path):.2f}")

# Ghi file nếu cần
with open("pathnames_75prefix.txt", "w") as f:
    for p in pathnames:
        f.write(p + "\n")
