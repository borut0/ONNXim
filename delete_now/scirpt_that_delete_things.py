from pathlib import Path

log_file = "low.log"  # Change this to your log filename

remove_patterns = [
    "Vector idle =",
    "SA idle ="
]

path = Path(log_file)

with path.open("r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

with path.open("w", encoding="utf-8") as f:
    for line in lines:
        if not any(pattern in line for pattern in remove_patterns):
            f.write(line)

print(f"Cleaned {log_file}")