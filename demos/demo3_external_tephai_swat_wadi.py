import argparse

def main():
    p=argparse.ArgumentParser(description="External dataset preparation demo."); p.parse_args(); print("Prepare HAI/TEP/SWaT/WADI raw files, then run scripts/prepare_<dataset>.py.")

if __name__=="__main__": main()
