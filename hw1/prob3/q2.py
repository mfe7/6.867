import numpy as np
from prob3_helper import regressData as load_data
# import regressData as load_data

def main():
    regress_a_X, regress_a_y = load_data.regressAData()
    regress_b_X, regress_b_y = load_data.regressBData()
    regress_val_X, regress_val_y = load_data.validateData()
    print regress_a_X
    return

if __name__ == '__main__':
    main()