"""Run the whole assignment: python src/run_all.py"""
import runpy, os, sys
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
for t in ("task1_restoration_segmentation", "task2_features", "task3_detection", "task4_motion"):
    print(f"\n===== {t} =====")
    runpy.run_path(os.path.join(here, t + ".py"), run_name="__main__")
