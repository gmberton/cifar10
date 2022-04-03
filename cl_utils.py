
import re

def readlines(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    return lines

def writefile(filename, content):
    if type(content) == list:
        content = "".join(content)
    with open(filename, "w") as file:
        _ = file.write(content)

def get_whole_lines(pattern, lines):
    return [l for l in lines if re.search(pattern, l)]

def get_matches(pattern, lines):
    lines = get_whole_lines(pattern, lines)
    return [re.search(pattern, l).groups()[0] for l in lines]

