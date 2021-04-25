import os
import glob
import tqdm
import string

def readdata(path):
    all_data = []
    list_file = glob.glob(path+"/*/*/*.txt")
    for elem in list_file:
        with open(elem, 'r') as f:
            data = f.read()
            label = elem.split('/')[-2]
            all_data.append(
                {
                    'file_name': elem, 
                    'data': data, 
                    'label': label
                }
            )
    return all_data


if __name__ == '__main__':
    data_path= '/Users/hit.fluoxetine/Dataset/nlp/data_train/'
    all_data = readdata(data_path)
    for data in tqdm.tqdm(all_data):
        file_name = data['file_name']
        raw_data= data['data']
        raw_data= raw_data.replace('_', ' ')
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in raw_data]
        raw_data=  ''.join(stripped)
        raw_data= raw_data.lower()
        raw_data= raw_data.split()
        raw_data= ' '.join(raw_data)
        with open(file_name, 'w') as f:
            f.writelines(raw_data)
    