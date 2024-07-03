# import os
# import shutil
#
#
# def count_files_in_folders(directory):
#     # 获取目录中的所有文件夹
#     folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
#
#     # 用于存储文件夹及其文件数量的数组
#     folder_file_counts = []
#
#     # 遍历每个文件夹，并统计文件数量
#     for folder in folders:
#         folder_path = os.path.join(directory, folder)
#         files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#         folder_file_counts.append((folder, len(files)))
#
#     # 按照文件数量排序
#     sorted_folder_file_counts = sorted(folder_file_counts, key=lambda x: x[1], reverse=True)
#
#     return sorted_folder_file_counts
#
#
# def keep_top_folders(directory, sorted_folder_file_counts, num_to_keep=100):
#     # 获取要保留的文件夹列表
#     top_folders = [folder for folder, _ in sorted_folder_file_counts[:num_to_keep]]
#
#     # 删除其他文件夹
#     for folder in os.listdir(directory):
#         if folder not in top_folders:
#             folder_path = os.path.join(directory, folder)
#             shutil.rmtree(folder_path)
#
#
# if __name__ == "__main__":
#     directory_path = "../data/classified_cell/"
#     result = count_files_in_folders(directory_path)
#
#     # 取数量最多的前100个文件夹，并删除其他文件夹
#     keep_top_folders(directory_path, result, num_to_keep=100)
#
#     print("已删除其他文件夹。")

import os


def count_files_in_folders(directory):
    # 获取目录中的所有文件夹
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # 用于存储文件夹及其文件数量的数组
    folder_file_counts = []

    # 遍历每个文件夹，并统计文件数量
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        folder_file_counts.append((folder, len(files)))

    # 按照文件数量排序
    sorted_folder_file_counts = sorted(folder_file_counts, key=lambda x: x[1])

    return sorted_folder_file_counts


if __name__ == "__main__":
    directory_path = "../data/classified_cell/"
    result = count_files_in_folders(directory_path)

    # 打印结果
    for folder, file_count in result:
        print(f"文件夹 '{folder}' 中的文件数量: {file_count}")

