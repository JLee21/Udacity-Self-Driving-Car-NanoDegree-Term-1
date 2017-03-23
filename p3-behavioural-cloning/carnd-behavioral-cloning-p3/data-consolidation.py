'''
THE GOAL

1.) Take a peak at the folders that containt the driving data
2.) Consolidate all the CSV files into a single list
3.) We will NOT augment or trim down any of th data in this script

Format of list
[
    [part1, part2, ..., part7]
    <many rows here>
    [part1, part2, ..., part7]
]

'''

from os import walk, path
from time import time
from csv import reader


def consolidate(path_to_folders, print_verbose=True):

    path_to_files = []
    lines = []

    if print_verbose:
        for a, b, c in walk(path_to_folders):
            print('root-------')
            print(a, '\n')
            print('\tdirs-------')
            for dir in b:
                # Skip the IMG folders
                if 'IMG' in dir: continue
                # Create a complete path to each folder
                print('\t\t', dir)
            print('\t\t\tfiles-------')
            print('\t\t\tfile count = {}'.format(len(c)))
            [print('\t\t\t\t', (file)) for file in c if 'csv' in file]
            file = [file for file in c if 'csv' in file]

    # Create list of full csv file paths
    for a, b, c in walk(path_to_folders):
        for file in c:
            if 'csv' in file:
                path_to_files.append(path.join(a, file))

    # Append lines of every CSV file to the list lines
    for file in path_to_files:
        folder_to_parse = file.split('\\')[-2]
        print('\nParsing \t\t==> {}'.format(folder_to_parse))

        start_time = time()
        with open(file, 'r', encoding='UTF-8') as csvfile:
            csv_reader = reader(csvfile)
            next(csv_reader, None)  # skip header
            for line in csv_reader:

                # create specific folder
                # so line[0] will be like this 'clockwise\\center.png'
                orig = line[0]
                print('orig', orig)
                new = folder_to_parse + '\\' + orig.split('\\')[-1]
                print('folder_to_parse', folder_to_parse)
                print('new', new)
                line[0] = new
                print('line ', line)
                break
                lines.append(line)
        print('\'lines\' length \t\t==>', len(lines))

        time_in_ms = (time()-start_time)*1000
        print('Done! Elapsed time(ms) \t==> {0:5.3f}'.format(time_in_ms))

    return lines

# if __name__ == '__main__':
#     lines = consolidate(r'D:\SDC\p3-Behavioural-Cloning\p3-behavorial-cloning\ricky')
#     print(len(lines))
