def get_line_number(contents, elem):
    """
    获取匹配项所在的行号
    :param contents: 匹配全文
    :param elem: 匹配项
    :return: 行号起止值(从0开始)
    """
    start_lineno = contents[:elem.span()[0]].count('\n')
    end_lineno = contents[:elem.span()[1]].count('\n')

    return start_lineno, end_lineno


def get_content_dict(contents):
    """
    功能：获取源码文件中行号对应的内容的dict
    参数： contents 是删除注释后的内容
    返回:contents_dict 行号对应的内容
    """
    contents_dict = {}
    contents_list = contents.splitlines()
    num = 1
    for line in contents_list:
        if not line:
            line = "\n"
        contents_dict[num] = line
        num = num + 1
    return contents_dict
