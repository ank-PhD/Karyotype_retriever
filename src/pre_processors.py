__author__ = 'Andrei'

from csv import reader as rdr

def get_centromeres():
    """
    Reads the centromere location from teh static_data cytoBand.txt file and returns it as a dict
    mapping each chromosome to its centromere location

    :return: dict {chr#: centromere location (kb)}
    """
    collector_list = []
    with open('static_data/cytoBand.txt') as fle:
        current_reader = rdr(fle, delimiter='\t')
        for line in current_reader:
            if line[-1] == 'acen' and 'p' in line[-2] and line[0][3:].isdigit():
                collector_list.append((int(line[0][3:]), int(line[2])/(10**3)))

    return dict(collector_list)



if __name__ == "__main__":
    print get_centromeres()

# TODO: convert the segmental data to point gain data
