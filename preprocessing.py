import numpy as np
from collections import defaultdict
import xlrd
file_name = 'file.xlsx'
file = xlrd.open_workbook(file_name)
sheet = file.sheet_by_name("Sheet1") 
col_value1 = sheet.col_values(0)
all_keys=col_value1
for file in all_keys:
    file_name=file[:5]
    with open('no_preprocessing_mutation_pdb/'+file, 'r') as f:
        data = f.read().strip().split('\n') 
        with open('./mutation_pdb/'+file+'_mutation.pdb',"w") as f:
            for a in data:
                if a[:4]=='ATOM':
                    b=a[23:26]
                    b=int(b)
                    c=file[7:-1]
                    c=int(c)
                    if c-3<=b<=c+3:                        
                         f.write(a+'\n')