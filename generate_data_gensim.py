import os
import cliport
import subprocess

CLIPORT_PATH = cliport.__path__[0]
GENSIM_PATH = os.path.dirname(CLIPORT_PATH)
DEMO_PATH = os.path.join(CLIPORT_PATH, "demos.py")
DATA_PATH = os.path.join(GENSIM_PATH, "data")

os.environ["GENSIM_ROOT"] = GENSIM_PATH

print(f"data collection script path: {DEMO_PATH}")

if __name__ == "__main__":
    n = 40
    tasks = [
        "align_balls_in_colored_boxes",
        "align_balls_in_colored_zones",
        "align_blocks_on_line",
        "align_boxes_on_line",
    ]
    for task in tasks:
        subprocess.run(["python", DEMO_PATH, f"task={task}", f"n={n}", "mode=train"])
