# EDA

- labels vs metadata:
    - metadata for 9% of patients (378 / 4178)
    - for overlapping:
        - CP

no link between labels and metadata (although there are overlapping patien ids)
- scan ids in metadata are not in the dataset

- n_slice:
    - between 16 and 690
    - avg 98, std 75.4
    - 257 covers 95%

## Check all labels (without checking metadata overlap)

- labels (4178 total):
    - In terms of observations:
        - 37% 1556 CP
        - 37% 1544 NCP
        - 26% 1078 Normal
    - In terms of slices:
        - 39% 159702 CP
        - 38% 156071 NCP
        - 23% 95756 Normal

- 2742 Unique patient ids. No patient with more then 1 label

- labels (unique patient ids):
    - 35% 964 CP
    - 34% 929 NCP
    - 31% 849 Normal

## Check labels with metadata only

- 378 labels
    - 51% CP
    -  2% NCP
    - 47% Normal

- 276 Unique patients:
    - 36% CP
    - 5% NCP
    - 59% Normal