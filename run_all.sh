available_datasets=(\
    "WMT19 - zh2en" "WMT19 - en2zh" "CSL - ctg" \
    "CSL - dcp" "CSL - ts" "Title2Event" "AFQMC" \
    "C3 - d" "C3 - m" "NCR - xdw" "CMRC" \
    "VCSum - short" "math23k" "math401" \
    "LogiQA" "CHiD" "DRCD" "NCR - gs" "NCR - wyw"\
    "MSRA - NER"\
)

#available_datasets=("math23k" "math401")
model="gpt-3.5-turbo"
sample_num=1            # comment this line to use all the samples
n_shot=0                # modify this for few-shot learning, but be aware of length constraint
# subset=""             # specify the subset; default values are in "defaults.py"
# icl_sample_file=""    # specify the icl file;default values are in "defaults.py"

for d in "${available_datasets[@]}"; do
    python main.py\
        -m "${model}"\
        -d "${d}"\
        -n "${sample_num}"\
        -ns "${n_shot}"
done

# if you want to specify the subset and icl_sample_file
#python main.py\
    #-m ${model}\
    #-d ${dataset}\
    #-n ${sample_num}\
    #-ns ${n_shot}\
    #-s ${subset}\
    #-f ${icl_sample_file}
