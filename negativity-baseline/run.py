"""Convenience entry: python run.py --quick"""
from quantneg.experiment import run, parse_args
if __name__=='__main__':
    run(parse_args())
