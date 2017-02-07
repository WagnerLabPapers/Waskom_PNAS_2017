# High-level commans for executing analysis

## Preprocessing, de-noising, and registration

### Experiment 1: Task
```
run_fmri.py -s dots_subjects -w preproc model reg -e dots \
            -regspace epi -unsmoothed -residual -timeseries \
            -n 14
```

### Experiment 2: Task
```
run_fmri.py -s sticks_subjects -w preproc model reg -e sticks \
            -regspace epi -unsmoothed -residual -timeseries \
            -n 12
```

### Experiment 2: Rest
```
run_fmri.py -s sticks_subjects -w preproc model reg -e rest \
            -regspace epi -regexp sticks -unsmoothed -residual -timeseries \
            -n 12
```

## ROI Extraction

```
bash run_script.sh roi_cache dots 'ifs mfc'
bash run_script.sh roi_cache sticks 'ifs mfc'
bash run_script.sh roi_cache rest 'ifs mfc'
```

## ROI Analysis

### Multivariate decoding analyses
```
bash run_script.sh decoding_analysis dots ifs
bash run_script.sh decoding_analysis dots mfc
bash run_script.sh decoding_analysis sticks ifs
bash run_script.sh decoding_analysis sticks mfc
```

### Analyses of context preference spatial organization
```
bash run_script.sh spatial_analysis dots ifs
bash run_script.sh spatial_analysis sticks ifs
```

### Analyses of subnetwork organization
```
bash run_script.sh correlation_analysis dots ifs
bash run_script.sh correlation_analysis sticks ifs
bash run_script.sh correlation_analysis rest ifs
```

### Post-processing of individual subject results
```
python compile_data.py
```
