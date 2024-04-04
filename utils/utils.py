import os


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_file_if_not_exists(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w').close()


def is_in_range(x, min, max):
    return x >= min and x <= max


def max_of_selected_columns(row, columns):
    selected = [row[column_name_or_index] for column_name_or_index in columns]
    return max(selected)


def two_columns_equal(row, column1, column2):
    if row[column1] == row[column2]:
        return 1.0
    return 0.0
