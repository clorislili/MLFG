atacseq_data = pd.read_csv(DATADIR + "brca_brca_peak_Log2Counts_dedup", sep='\t')
atacseq_data.head()
methylation_data = methylation_data_full[methylation_data_full.columns[methylation_data_full.columns.isin(atacseq_data.columns)]]
methylation_data.to_csv(DATADIR + "TCGA-BRCA.methylation.tsv", sep="\t")
methylation_data = pd.read_csv(DATADIR + "TCGA-BRCA.methylation.tsv", sep='\t').dropna()
print(methylation_data.shape)
methylation_data.head()
gene_mapping = pd.read_csv(DATADIR + "brca_brca_peak.probeMap", sep='\t')
gene_mapping = gene_mapping[ ~gene_mapping['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
gene_mapping = gene_mapping.sort_values(['chrom', 'chromStart']).drop_duplicates() # sort so we can interleave negatives
gene_mapping.head()
gene_mapping_methylation = pd.read_csv(DATADIR + "illuminaMethyl450_hg38_GDC", sep='\t')
gene_mapping_methylation = gene_mapping_methylation[ ~gene_mapping_methylation['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
gene_mapping_methylation = gene_mapping_methylation.sort_values(['#id']).drop_duplicates() # sort so we can interleave negatives
gene_mapping_methylation = gene_mapping_methylation[ ~gene_mapping_methylation['gene'].isin(["."]) ]
gene_mapping_methylation.head()
methylation = methylation_data.set_index('Composite Element REF').join(gene_mapping_methylation.set_index('#id'))
methylation= methylation.sort_values(['chrom', 'chromStart']).dropna()
methylation = methylation.drop(columns=['gene','strand'])
methylation
atacseq = atacseq_data.set_index('sample').join(gene_mapping.set_index('id'))
atacseq = atacseq[atacseq.columns[atacseq.columns.isin(methylation.columns)]]
atacseq = atacseq.sort_values(['chrom', 'chromStart']).dropna()
atacseq
atacseq.index = pd.IntervalIndex.from_arrays(atacseq['chromStart'],atacseq['chromEnd'],closed='both',name='chromRange')
atacseq = atacseq.sort_index(axis=1)
atacseq
methylation['chromRange'] = ","
methylation = methylation.set_index('chromStart')
methylation = methylation.sort_index(axis=1)
methylation

from tqdm import tqdm
import re 

chromRange = atacseq.index
count=1
chromRange_copy = [] # when using .iterrows, it generate a copy of the dataFrame, when altering "row, "the original data will not be changed. So we need to generate a list and replace the column in the end.
for i, row in tqdm(methylation.iterrows()):
  bool_list = chromRange.contains(i)
  chrom_match = np.where(atacseq.chrom==row.chrom, True, False)
  both_true = bool_list & chrom_match

  if True in bool_list:   
    interval = chromRange[np.where(both_true)]
    interval = re.findall("\[(\d+.0, \d+.0)\]",str(interval))
    if len(interval) > 0:
      chromRange_copy.append(interval)
    else:
      chromRange_copy.append(",")
  else:
    chromRange_copy.append(",")
    # print('2',i)
  #print(chromRange_copy)
 #though row has changed to range, but the dataframe methylation still can not be changed
  count+=1

methylation['chromRange'] = chromRange_copy
methylation.to_csv(DATADIR+'methylation_matched.csv')
methylation_data = pd.read_csv(DATADIR + "methylation_matched.csv")
methylation_data.head()
methylation_data = methylation_data[ ~methylation_data['chromRange'].isin([","]) ]
print(methylation_data.shape)
methylation_data[:10]
parsed = methylation_data['chromRange'].str.replace(r'(\')','')
methylation_data['chromRange'] = parsed
atacseq = atacseq.reset_index()
atacseq['chromRange'] = atacseq.chromRange.astype("|S")
methylation_data['chromRange'] = methylation_data.chromRange.astype("|S")
data = pd.merge(
    atacseq,
    methylation_data,
    how="inner",
    on=["chromRange","chrom"],
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)
data = data.drop(columns=['chromEnd_x','chromStart_x','chromStart_y'])
data = data.reindex(sorted(data.columns), axis=1)
data.to_csv(DATADIR+'data.csv') 

import numpy as np
import pandas as pd
from tqdm import tqdm
data = pd.read_csv("./data.csv")

np_data = []
np_data.append(np.array(data.columns))
np_data.append(np.array(data.values))
np_data = np.array(np_data)

train_x = []
train_y = []
for i in tqdm(range(125828)):
    chrom1 = np_data[1][i][-1:]
    s = str(chrom1[0]).split('[')[1].split(']')[0].split(' ')[0][:-3]
    e = str(chrom1[0]).split('[')[1].split(']')[0].split(' ')[1][:-2]


    chrom2 = np_data[1][i][-2:-1]
    chrom3 = np_data[1][i][-3:-2][0][3:]

    for j in range(1,85,2):

        temp = []
        y = []

        temp.append(np_data[0][j][:-2])
        temp1 = np_data[1][i][j:j+1]

        temp.append(float(temp1))
        temp2 = np_data[1][i][j+1:j+2]
        
        temp.append(int(chrom3))
        temp.append(int(chrom2))
        temp.append(int(str(s)))
        temp.append(int(str(e)))

        
        y.append(np_data[0][j][:-2])
        y.append(float(temp2))

        train_y.append(y)
        train_x.append(temp)

train_x=pd.DataFrame(columns=['sample_name','x','chrm', 'end','range_start', 'range_end'],data=train_x)
train_x.to_csv('./train_x.csv')
train_y=pd.DataFrame(columns=['sample_name','y'],data=train_y)
train_y.to_csv('./train_y.csv')