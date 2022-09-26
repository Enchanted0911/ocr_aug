def write_str_list_to_txt(str_list, file_path):
    with open(file_path, 'a+', encoding='utf-8') as f:
        for data in str_list:
            f.write(data + '\n')
        f.close()
