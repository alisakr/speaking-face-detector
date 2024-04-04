import argparse
import pandas as pd
import os
import sys
sys.path.append('./')

from utils import create_file_if_not_exists


def main():
    '''
    Combine csvs together as dataframes and output a single csv
    csvs should have same headers
    example input csvs: anchorbro_1_12.csv,bay_bridge_13_33.csv,powell_anchorwoman.csv,saagar_0_10.csv,test_of_kyle_matrix2.csv,test_of_kyle_matrix.csv,tnt.csv,yeahmad_0_6.csv,yeahmad_73_99.csv,
    
    '''
    parser = argparse.ArgumentParser(description='combine csvs together and append to an output csv')
    parser.add_argument('--input_csvs', default='', type=str, help='comma separated csvs to combine')
    parser.add_argument('--output_csv', type=str, help='output csv, if already exists, will append to it, unless overwrite is set')
    parser.add_argument('--overwrite', default=False, type=bool, help='overwrite output csv')
    parser.add_argument('--skipped_columns', default='', type=str, help='columns to skip, should not be in output csv')
    args = parser.parse_args()
    input_csvs = args.input_csvs.split(',')
    output_csv = args.output_csv
    overwrite = args.overwrite
    if overwrite is True:
        print("overwriting")
    else:
        print("appending, not overwriting")
    final_df = None
    if os.path.exists(output_csv):
        if not overwrite:
            final_df = pd.read_csv(output_csv)
    else:
        create_file_if_not_exists(output_csv)
    
    for csv in input_csvs:
        df = pd.read_csv(csv)
        if args.skipped_columns != '':
            df.drop(columns=args.skipped_columns.split(','), inplace=True)
        if final_df is None:
            final_df = df
        else:
            if not final_df.columns.equals(df.columns):
                print("columns not equal")
                print(final_df.columns)
                print(df.columns)
                raise ValueError("columns not equal")
            final_df = pd.concat([final_df, df], ignore_index=True)
    final_df.to_csv(output_csv, index=False)


main()